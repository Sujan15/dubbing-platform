# hindi-dubbing-platform/backend/services/speaker_identity.py

"""Speaker identity layer on top of pyannote diarization.

pyannote labels are a per-run clustering; they say nothing about who a speaker is in another
episode. This module adds what the labels lack:

* ``SpeakerEmbedder`` - one ECAPA-TDNN (SpeechBrain) embedding per *episode speaker*, computed
  from the same clean solo spans the clone reference uses (long, non-overlapped speech only).
* ``match_speakers`` - cosine matching of episode speakers against the characters' embedding bank
  with two calibrated thresholds: ``>= high`` -> suggested (still requires a human click),
  ``[review, high)`` -> suggested with a review flag, ``< review`` -> unassigned.
* ``speech evidence policy`` - a speaker with less than ``MIN_SEGMENT_IDENTITY_DURATION`` of clean
  solo speech is **unassigned** whatever the similarity: a 0.4 s "hmm" is not identity evidence.
* ``attribution_confidence`` - per dialogue block, how well the words were covered by one
  diarization turn; short / contested blocks are flagged for review instead of trusted.

Nothing here assigns a character automatically. The output is a suggestion that the Human Review
workstation shows next to each local speaker.

Thresholds are configuration (``SPEAKER_MATCH_*``) and must be calibrated per catalogue of
recordings; ``calibrate`` returns the data needed to pick them from a labelled set.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from backend.app.config import Settings
from backend.services.media import read_audio

log = logging.getLogger("dubbing.identity")


# =============================================================================== embeddings
class SpeakerEmbedder:
    """ECAPA-TDNN speaker verification embeddings (SpeechBrain), loaded lazily and released after use."""

    def __init__(self, settings: Settings, device: str):
        self.s = settings
        self.device = device
        self._model = None

    @staticmethod
    def available() -> bool:
        try:
            import speechbrain  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self):
        if self._model is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                from speechbrain.utils.fetching import LocalStrategy
            except ImportError:  # speechbrain < 1.0
                from speechbrain.pretrained import EncoderClassifier
            # self._model = EncoderClassifier.from_hparams(
            #     source=self.s.speaker_embedding_model,
            #     savedir=os.fspath(self.s.models_dir / "speechbrain" / self.s.speaker_embedding_model.replace("/", "--")),
            #     run_opts={"device": self.device})
            self._model = EncoderClassifier.from_hparams(
                source=self.s.speaker_embedding_model,
                savedir=os.fspath(
                    self.s.models_dir
                    / "speechbrain"
                    / self.s.speaker_embedding_model.replace("/", "--")
                ),
                run_opts={"device": self.device},
                local_strategy=LocalStrategy.COPY,
            )
        return self._model

    def embed_file(self, path: Path, sample_rate: int = 16000) -> np.ndarray:
        import torch

        data, sr = read_audio(path)
        mono = data.mean(axis=1)
        if sr != sample_rate:
            import torchaudio
            mono = torchaudio.functional.resample(torch.from_numpy(mono), sr, sample_rate).numpy()
        wav = torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32))[None]
        with torch.inference_mode():
            emb = self._load().encode_batch(wav).squeeze().cpu().numpy().astype(np.float32)
        return unit(emb)

    def unload(self) -> None:
        self._model = None


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# =============================================================================== matching
@dataclass
class Candidate:
    character_id: str
    similarity: float


@dataclass
class Suggestion:
    local_speaker_id: str
    status: str                                   # suggested | review | unassigned
    character_id: Optional[str] = None
    confidence: Optional[float] = None
    reason: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    speech_seconds: float = 0.0

    def public(self) -> dict:
        return {"local_speaker_id": self.local_speaker_id, "status": self.status, "character_id": self.character_id,
                "confidence": None if self.confidence is None else round(self.confidence, 4), "reason": self.reason,
                "speech_seconds": round(self.speech_seconds, 2),
                "candidates": [{"character_id": c.character_id, "similarity": round(c.similarity, 4)}
                               for c in self.candidates[:5]]}


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return a @ b.T


def match_speakers(episode: dict[str, np.ndarray], speech_seconds: dict[str, float],
                   characters: dict[str, np.ndarray], settings: Settings) -> list[Suggestion]:
    """Suggest a canonical character for every episode speaker (or UNASSIGNED with the reason).

    ``episode``: local_speaker_id -> embedding; ``characters``: character_id -> (n, d) vectors.
    The same character may be suggested for two local speakers (a false split); the review UI
    shows both and the user merges or rejects - this function never merges.
    """
    high, review = settings.speaker_match_high_threshold, settings.speaker_match_review_threshold
    min_secs = settings.min_segment_identity_duration
    out: list[Suggestion] = []
    for spk in sorted(set(episode) | set(speech_seconds)):
        secs = float(speech_seconds.get(spk, 0.0))
        emb = episode.get(spk)
        if emb is None:
            out.append(Suggestion(spk, "unassigned", reason="no embedding (no clean solo speech)", speech_seconds=secs))
            continue
        if secs < min_secs:
            out.append(Suggestion(spk, "unassigned", speech_seconds=secs,
                                  reason=f"only {secs:.1f}s of clean speech (< MIN_SEGMENT_IDENTITY_DURATION={min_secs:g}); "
                                         "too little evidence for an identity match"))
            continue
        if not characters:
            out.append(Suggestion(spk, "unassigned", reason="no characters with voice evidence yet", speech_seconds=secs))
            continue
        cands = []
        for cid, vecs in characters.items():
            sim = float(cosine_matrix(emb[None], vecs).max())
            cands.append(Candidate(cid, sim))
        cands.sort(key=lambda c: c.similarity, reverse=True)
        best = cands[0]
        runner = cands[1].similarity if len(cands) > 1 else -1.0
        if best.similarity >= high and best.similarity - runner >= 0.05:
            out.append(Suggestion(spk, "suggested", best.character_id, best.similarity,
                                  "strong match; confirm in review", cands, secs))
        elif best.similarity >= review:
            why = "ambiguous between candidates" if best.similarity - runner < 0.05 else "match below the high threshold"
            out.append(Suggestion(spk, "review", best.character_id, best.similarity, f"{why}; listen before confirming",
                                  cands, secs))
        else:
            out.append(Suggestion(spk, "unassigned", None, best.similarity,
                                  "no character is similar enough; create or pick one manually", cands, secs))
    return out


def calibrate(pairs: list[tuple[float, bool]]) -> dict:
    """Threshold candidates from labelled (similarity, same_speaker) pairs: precision/recall per cut."""
    if not pairs:
        return {"pairs": 0, "table": []}
    sims = np.array([p[0] for p in pairs]); same = np.array([p[1] for p in pairs], dtype=bool)
    table = []
    for t in np.round(np.arange(0.3, 0.96, 0.05), 2):
        pred = sims >= t
        tp = int((pred & same).sum()); fp = int((pred & ~same).sum()); fn = int((~pred & same).sum())
        table.append({"threshold": float(t), "precision": round(tp / max(tp + fp, 1), 3),
                      "recall": round(tp / max(tp + fn, 1), 3), "false_accepts": fp, "false_rejects": fn})
    return {"pairs": len(pairs), "table": table}


# =============================================================================== attribution
def attribution_confidence(block: dict, turns: list[dict]) -> tuple[float, list[str]]:
    """How much of a dialogue block is covered by the turn of its own speaker, and competing overlap."""
    s, e, spk = block["start"], block["end"], block["speaker"]
    dur = max(e - s, 1e-3)
    own = sum(max(0.0, min(e, t["end"]) - max(s, t["start"])) for t in turns if t["speaker"] == spk)
    other = sum(max(0.0, min(e, t["end"]) - max(s, t["start"])) for t in turns if t["speaker"] != spk)
    coverage = min(1.0, own / dur)
    contested = min(1.0, other / dur)
    conf = max(0.0, coverage - 0.5 * contested)
    flags = []
    if coverage < 0.5:
        flags.append("speaker turn covers less than half of the line")
    if contested > 0.3:
        flags.append("another speaker overlaps this line")
    return round(conf, 3), flags
