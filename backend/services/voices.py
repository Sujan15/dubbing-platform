# hindi-dubbing-platform/backend/services/voices.py

"""Voice references: validation/normalisation, the historical default bank and voice resolution.

A voice reference is a short WAV paired with its transcript for IndicF5. It can come from:

* ``clone``   - clean speech of the speaker cut from the current video (pipeline ``voices`` stage)
* ``library`` - a transcript-bearing reference saved in the project's Speaker Library
* ``default`` - a historical file from the default voice bank (not used for active synthesis)

Resolution never substitutes silently: every fallback is recorded on the ``ResolvedVoice`` and
surfaces as a job warning and in ``voice_resolution.json``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from backend.app.config import Settings, TTSEngine
from backend.services.media import FFmpeg, MediaError, write_audio
from backend.services.storage import VOICE_ID_RE
from backend.services.tts.base import VoiceSpec

log = logging.getLogger("dubbing.voices")

HARD_MIN_SPEECH_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 120.0
_FRAME_SECONDS = 0.03


class VoiceSource(str, Enum):
    CLONE = "clone"            # reference cut from the current video (IndicF5)
    LIBRARY = "library"        # transcript-bearing project Speaker Library reference
    DEFAULT = "default"        # historical source; no longer supported for synthesis
    CAST = "cast"              # approved Voice Library voice (Parler preset or IndicF5 clone)
    CHARACTER = "character"    # the canonical character's active, approved voice profile


class ReferenceAudioError(ValueError):
    pass


class VoiceResolutionError(RuntimeError):
    pass


@dataclass
class ReferenceReport:
    duration_seconds: float
    speech_seconds: float
    sample_rate: int
    channels: int
    peak_dbfs: float
    warnings: list[str] = field(default_factory=list)


def analyze_reference(path: Path) -> ReferenceReport:
    """Measure a decodable WAV: duration, active-speech seconds (energy based), peak level."""
    try:
        data, sr = sf.read(os.fspath(path), dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise ReferenceAudioError(f"Audio is not decodable: {exc}") from exc
    channels = data.shape[1]
    mono = data.mean(axis=1)
    duration = mono.size / float(sr) if sr else 0.0
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    peak_db = 20 * np.log10(peak + 1e-9)
    win = max(1, int(sr * _FRAME_SECONDS))
    n = mono.size // win
    speech = 0.0
    if n and peak > 1e-5:
        rms = np.sqrt(np.mean(mono[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
        rel = 20 * np.log10(rms / peak)
        active = (rel > -35.0) & (20 * np.log10(rms) > -55.0)
        speech = float(active.sum() * win / sr)
    return ReferenceReport(round(duration, 3), round(speech, 3), int(sr), int(channels), round(float(peak_db), 2))


def normalize_reference(ff: FFmpeg, src: Path, dst: Path, settings: Settings) -> ReferenceReport:
    """Convert any decodable audio into the TTS reference format and validate it.

    Output: mono, 16-bit PCM, ``SPEAKER_REFERENCE_SAMPLE_RATE``, peak-normalised to -1 dBFS.
    ``dst`` is only replaced when every check passes.
    """
    tmp = dst.with_name(f"{dst.stem}.validating.wav")
    try:
        try:
            ff.resample(src, tmp, settings.speaker_reference_sample_rate, 1, codec="pcm_s16le")
        except MediaError as exc:
            raise ReferenceAudioError(f"Audio could not be decoded: {str(exc).splitlines()[-1][:200]}") from exc
        report = analyze_reference(tmp)
        if report.duration_seconds > MAX_REFERENCE_SECONDS:
            raise ReferenceAudioError(
                f"Reference is {report.duration_seconds:.0f}s; trim it to 15-30 s of clean speech "
                f"(maximum {MAX_REFERENCE_SECONDS:.0f}s).")
        if report.peak_dbfs < -50.0:
            raise ReferenceAudioError("Reference audio is silent.")
        if report.speech_seconds < HARD_MIN_SPEECH_SECONDS:
            raise ReferenceAudioError(
                f"Only {report.speech_seconds:.1f}s of speech detected; at least {HARD_MIN_SPEECH_SECONDS:.0f}s "
                "is required (15-20 s recommended).")
        if report.speech_seconds < settings.min_speaker_reference_seconds:
            report.warnings.append(
                f"Reference has {report.speech_seconds:.1f}s of speech, below MIN_SPEAKER_REFERENCE_SECONDS="
                f"{settings.min_speaker_reference_seconds:g}; cloning will be less faithful.")
        data, sr = sf.read(os.fspath(tmp), dtype="float32", always_2d=True)
        peak = float(np.max(np.abs(data)))
        data = data * (0.891 / peak)
        write_audio(dst, data, sr, subtype="PCM_16")
        return report
    finally:
        tmp.unlink(missing_ok=True)


# =============================================================================== default bank
@dataclass
class DefaultVoice:
    voice_id: str
    label: str
    description: str
    gender: Optional[str]
    duration_seconds: float
    speech_seconds: float
    sample_rate: int
    valid: bool
    problem: Optional[str]
    path: Path

    def public(self) -> dict:
        d = asdict(self)
        d.pop("path")
        return d


class DefaultVoiceBank:
    """Read-only bank of fallback voices: ``<dir>/<id>.wav`` plus an optional ``<id>.json`` sidecar
    (``{"label": ..., "description": ..., "gender": ...}``). Validation results are cached per
    file size/mtime, so adding a file is picked up without a restart."""

    def __init__(self, settings: Settings):
        self.dir = Path(settings.default_voices_dir)
        self._cache: dict[str, tuple[tuple, DefaultVoice]] = {}
        self._lock = threading.Lock()

    def list(self, include_invalid: bool = False) -> list[DefaultVoice]:
        if not self.dir.is_dir():
            return []
        voices = []
        for wav in sorted(self.dir.glob("*.wav")):
            if not VOICE_ID_RE.fullmatch(wav.stem):
                continue
            v = self._load(wav)
            if v.valid or include_invalid:
                voices.append(v)
        return voices

    def get(self, voice_id: str) -> Optional[DefaultVoice]:
        if not voice_id or not VOICE_ID_RE.fullmatch(voice_id):
            return None
        wav = self.dir / f"{voice_id}.wav"
        if not wav.is_file():
            return None
        v = self._load(wav)
        return v if v.valid else None

    def _load(self, wav: Path) -> DefaultVoice:
        sidecar = wav.with_suffix(".json")
        st = wav.stat()
        sig = (st.st_size, st.st_mtime_ns, sidecar.stat().st_mtime_ns if sidecar.is_file() else 0)
        with self._lock:
            hit = self._cache.get(wav.stem)
        if hit and hit[0] == sig:
            return hit[1]
        meta: dict = {}
        if sidecar.is_file():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                log.warning("Ignoring unreadable default-voice metadata %s: %s", sidecar.name, exc)
        problem = None
        try:
            rep = analyze_reference(wav)
            if rep.speech_seconds < HARD_MIN_SPEECH_SECONDS:
                problem = f"only {rep.speech_seconds:.1f}s of speech"
        except ReferenceAudioError as exc:
            rep, problem = ReferenceReport(0, 0, 0, 0, -120), str(exc)
        voice = DefaultVoice(
            voice_id=wav.stem, label=str(meta.get("label") or wav.stem.replace("_", " ").title()),
            description=str(meta.get("description") or ""), gender=meta.get("gender"),
            duration_seconds=rep.duration_seconds, speech_seconds=rep.speech_seconds, sample_rate=rep.sample_rate,
            valid=problem is None, problem=problem, path=wav)
        if problem:
            log.warning("Default voice %s is not usable: %s", wav.name, problem)
        with self._lock:
            self._cache[wav.stem] = (sig, voice)
        return voice


# =============================================================================== resolution
@dataclass
class ResolvedVoice:
    speaker_id: str
    display_name: str
    requested_source: VoiceSource
    resolved_source: VoiceSource
    reference_path: Optional[Path]
    reference_seconds: float
    library_speaker_id: Optional[str] = None
    default_voice_id: Optional[str] = None
    fallback_applied: bool = False
    warnings: list[str] = field(default_factory=list)
    # v3: engine-aware
    engine: Optional[TTSEngine] = None
    spec: Optional[VoiceSpec] = None
    voice_id: Optional[str] = None
    character_id: Optional[str] = None
    voice_profile_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.spec is None:
            if self.engine is None or self.engine == TTSEngine.XTTS:
                raise VoiceResolutionError("A supported TTS engine must be selected explicitly.")
            self.spec = VoiceSpec(engine=self.engine, reference_path=self.reference_path, label=self.display_name)
        self.engine = self.spec.engine
        if self.reference_path is None:
            self.reference_path = self.spec.reference_path

    def public(self) -> dict:
        d = asdict(self)
        d["requested_source"] = self.requested_source.value
        d["resolved_source"] = self.resolved_source.value
        d["engine"] = self.engine.value if self.engine else None
        d.pop("reference_path")
        spec = d.pop("spec") or {}
        d["reference_transcript"] = spec.get("reference_transcript")
        d["voice_prompt"] = spec.get("voice_prompt")
        d["generation_parameters"] = spec.get("generation_parameters") or {}
        d["seed"] = spec.get("seed")
        return d


# (voice_id) -> (VoiceSpec, record) for approved + active Voice Library voices, else None
VoiceLibraryLookup = Callable[[str], Optional[tuple[VoiceSpec, object]]]


@dataclass
class CharacterVoice:
    """What a character's active, approved voice profile resolves to."""
    character_id: str
    character_name: str
    voice_profile_id: str
    voice_source: VoiceSource          # cast | clone (voice library) | library | default
    engine: TTSEngine
    voice_asset_id: Optional[str]


# (character_id) -> CharacterVoice or None (no character / no approved active profile)
CharacterLookup = Callable[[str], Optional[CharacterVoice]]


# (library_speaker_id) -> (reference path, speech seconds, display name, transcript) or None
LibraryLookup = Callable[[str], Optional[tuple[Path, float, str, Optional[str]]]]


@dataclass
class VoiceRequest:
    speaker_id: str
    display_name: str
    voice_source: VoiceSource = VoiceSource.CLONE
    library_speaker_id: Optional[str] = None
    default_voice_id: Optional[str] = None
    # v3
    engine: Optional[TTSEngine] = None          # clone: IndicF5 only
    voice_id: Optional[str] = None              # cast: Voice Library voice
    voice_prompt: Optional[str] = None          # one-shot Parler prompt, never a project assignment
    character_id: Optional[str] = None          # character: canonical character


def resolve_voice(req: VoiceRequest, clone_ref: Optional[Path], clone_seconds: float,
                  library: Optional[LibraryLookup], bank: DefaultVoiceBank, settings: Settings,
                  voice_library: Optional[VoiceLibraryLookup] = None, characters: Optional[CharacterLookup] = None,
                  clone_transcript: Optional[str] = None) -> ResolvedVoice:
    """Turn a speaker's voice selection into one concrete ``VoiceSpec`` (or raise). Never substitutes silently."""
    base = {"speaker_id": req.speaker_id, "display_name": req.display_name, "requested_source": req.voice_source}

    if req.voice_source == VoiceSource.CHARACTER:
        cv = characters(req.character_id) if (characters and req.character_id) else None
        if cv is None:
            raise VoiceResolutionError(
                f"{req.display_name}: character '{req.character_id}' has no approved active voice profile.")
        inner = VoiceRequest(speaker_id=req.speaker_id, display_name=req.display_name, voice_source=cv.voice_source,
                             library_speaker_id=cv.voice_asset_id if cv.voice_source == VoiceSource.LIBRARY else None,
                             default_voice_id=cv.voice_asset_id if cv.voice_source == VoiceSource.DEFAULT else None,
                             voice_id=cv.voice_asset_id if cv.voice_source in (VoiceSource.CAST, VoiceSource.CLONE) else None,
                             engine=cv.engine)
        if inner.voice_source == VoiceSource.CLONE:
            inner.voice_source = VoiceSource.CAST  # a character clone lives in the Voice Library, never in one episode
        rv = resolve_voice(inner, None, 0.0, library, bank, settings, voice_library, None)
        if rv.fallback_applied:
            raise VoiceResolutionError(f"{cv.character_name}: permanent voice profile could not be resolved.")
        rv.requested_source = VoiceSource.CHARACTER
        rv.character_id, rv.voice_profile_id = cv.character_id, cv.voice_profile_id
        return rv

    if req.voice_source == VoiceSource.CAST:
        if req.voice_prompt and not req.voice_id:
            if TTSEngine.PARLER not in settings.enabled_tts_engines:
                raise VoiceResolutionError(
                    f"{req.display_name}: Parler is not enabled for this one-shot voice prompt.")
            spec = VoiceSpec(engine=TTSEngine.PARLER, voice_prompt=req.voice_prompt, label=req.display_name)
            return ResolvedVoice(**base, resolved_source=VoiceSource.CAST, reference_path=None,
                                 reference_seconds=0.0, spec=spec)
        found = voice_library(req.voice_id) if (voice_library and req.voice_id) else None
        if not found:
            raise VoiceResolutionError(
                f"{req.display_name}: voice '{req.voice_id}' is not an approved, active Voice Library voice.")
        spec, rec = found
        return ResolvedVoice(**base, resolved_source=VoiceSource.CAST, reference_path=spec.reference_path,
                             reference_seconds=float(getattr(rec, "reference_speech_seconds", 0.0) or 0.0),
                             spec=spec, voice_id=req.voice_id)

    if req.voice_source == VoiceSource.LIBRARY:
        found = library(req.library_speaker_id) if (library and req.library_speaker_id) else None
        if not found:
            raise VoiceResolutionError(
                f"{req.display_name}: library speaker '{req.library_speaker_id}' is not in this project or has no "
                "validated reference audio.")
        path, seconds, _, transcript = found
        if not (transcript or "").strip():
            raise VoiceResolutionError(
                f"{req.display_name}: project library speaker '{req.library_speaker_id}' has no reference "
                "transcript; add the transcript before selecting this IndicF5 voice.")
        return ResolvedVoice(**base, resolved_source=VoiceSource.LIBRARY, reference_path=path,
                             reference_seconds=seconds, library_speaker_id=req.library_speaker_id,
                             spec=VoiceSpec(engine=TTSEngine.INDICF5, reference_path=path, label=req.display_name,
                                            reference_transcript=transcript))

    if req.voice_source == VoiceSource.DEFAULT:
        raise VoiceResolutionError(
            f"{req.display_name}: default voice assignments used the retired XTTS engine; select an approved "
            "Parler profile or a transcript-bearing IndicF5 reference.")

    # clone
    engine = req.engine
    if engine is None or engine == TTSEngine.XTTS:
        raise VoiceResolutionError(
            f"{req.display_name}: this saved clone assignment has no supported engine; explicitly select IndicF5 "
            "or an approved Parler profile.")
    if engine != TTSEngine.INDICF5:
        raise VoiceResolutionError(
            f"{req.display_name}: clone engine '{engine.value}' is not supported; Parler uses approved voice profiles.")
    if engine not in settings.enabled_tts_engines:
        raise VoiceResolutionError(f"{req.display_name}: clone engine '{engine.value}' is not enabled.")
    if engine == TTSEngine.INDICF5 and not (clone_transcript or "").strip():
        raise VoiceResolutionError(
            f"{req.display_name}: IndicF5 cloning needs the transcript of the reference speech; select a "
            "transcript-bearing reference or an approved Parler profile.")
    minimum = settings.min_speaker_reference_seconds
    if clone_ref is not None and clone_seconds >= minimum:
        return ResolvedVoice(**base, resolved_source=VoiceSource.CLONE, reference_path=clone_ref,
                             reference_seconds=clone_seconds,
                             spec=VoiceSpec(engine=engine, reference_path=clone_ref, label=req.display_name,
                                            reference_transcript=clone_transcript if engine == TTSEngine.INDICF5 else None))
    have = f"{clone_seconds:.1f}s" if clone_ref is not None else "no usable"
    if clone_ref is not None:
        msg = (f"{req.display_name}: cloning from only {clone_seconds:.1f}s of speech "
               f"(< MIN_SPEAKER_REFERENCE_SECONDS={minimum:g}); voice similarity may be lower.")
        return ResolvedVoice(**base, resolved_source=VoiceSource.CLONE, reference_path=clone_ref,
                             reference_seconds=clone_seconds, warnings=[msg],
                             spec=VoiceSpec(engine=engine, reference_path=clone_ref, label=req.display_name,
                                            reference_transcript=clone_transcript if engine == TTSEngine.INDICF5 else None))
    raise VoiceResolutionError(
        f"Speaker {req.display_name} has no usable reference audio; select a transcript-bearing library reference "
        "or an approved Parler voice profile.")
