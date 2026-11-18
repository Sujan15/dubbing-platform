# hindi-dubbing-platform/backend/services/tts/base.py

"""Unified TTS engine contract.

Every adapter (IndicF5 clone, Parler cast) implements ``BaseTTSEngine``. The pipeline
never branches on engine-specific conditions: it asks the registry for the engine named in the
resolved voice, hands it a ``VoiceSpec`` and gets float32 mono audio back.

``VoiceSpec`` carries *everything that can change the audio*: reference audio (by content hash),
reference transcript, voice description prompt, model id/revision and generation parameters. Its
``cache_fingerprint()`` is part of the per-line TTS cache key, so a changed prompt or checkpoint
can never return stale audio.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from backend.app.config import Settings, TTSEngine


class TTSEngineError(RuntimeError):
    pass


class EngineUnavailable(TTSEngineError):
    """The engine cannot run in this installation (missing package, licence, model or interpreter)."""


@dataclass(frozen=True)
class EngineCapabilities:
    engine: TTSEngine
    reference_conditioned: bool          # needs reference audio
    needs_reference_transcript: bool     # IndicF5 conditions on the transcript of the reference too
    prompt_conditioned: bool             # Parler: identity comes from a text description
    supports_seed: bool
    supports_speed: bool
    languages: tuple[str, ...]
    commercial_use: Optional[bool]       # None = check the licence yourself
    license: str


@dataclass(frozen=True)
class EngineMetadata:
    engine: TTSEngine
    model_id: str
    model_revision: Optional[str]
    sample_rate: int
    device: str
    isolated: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class VoiceSpec:
    """One concrete voice, independent of where it came from (clone / character / cast / default)."""
    engine: TTSEngine
    reference_path: Optional[Path] = None
    reference_transcript: Optional[str] = None
    voice_prompt: Optional[str] = None
    generation_parameters: dict = field(default_factory=dict)
    seed: Optional[int] = None
    voice_id: Optional[str] = None            # voice library record, when any
    label: str = ""

    def cache_fingerprint(self, reference_digest: Optional[str]) -> dict:
        """Deterministic description of the inputs that influence audio (reference by *content* hash)."""
        return {
            "engine": self.engine.value,
            "reference": reference_digest,
            "reference_transcript": self.reference_transcript,
            "voice_prompt": self.voice_prompt,
            "params": _canonical(self.generation_parameters),
            "seed": self.seed,
        }

    def public(self) -> dict:
        return {"engine": self.engine.value, "voice_id": self.voice_id, "label": self.label,
                "reference_transcript": self.reference_transcript, "voice_prompt": self.voice_prompt,
                "generation_parameters": _canonical(self.generation_parameters), "seed": self.seed,
                "reference_path": str(self.reference_path) if self.reference_path else None}


class BaseTTSEngine:
    """Contract shared by every adapter; the pipeline holds at most one loaded engine at a time."""

    engine: TTSEngine
    capabilities: EngineCapabilities
    sample_rate: int = 24000
    fingerprint: dict = {}                   # model-level inputs that change audio (joined with VoiceSpec)

    # ---- static checks (no model loaded) ----------------------------------------
    @classmethod
    def validate_configuration(cls, settings: Settings) -> list[str]:
        """Configuration problems that make the engine unusable (empty = fine)."""
        return []

    @classmethod
    def check_model_availability(cls, settings: Settings) -> dict:
        """{'available': bool, 'reason': str, 'model_id': ..., 'revision': ...} without downloading anything."""
        raise NotImplementedError

    @classmethod
    def get_capabilities(cls) -> EngineCapabilities:
        return cls.capabilities

    # ---- runtime -------------------------------------------------------------------
    def __init__(self, settings: Settings, device: str):
        self.settings = settings
        self.device = device

    def get_model_metadata(self) -> EngineMetadata:
        raise NotImplementedError

    def synthesize(self, text: str, spec: VoiceSpec, seed: int, speed: float = 1.0) -> np.ndarray:
        """float32 mono at ``self.sample_rate``; raise ``TTSEngineError`` on failure."""
        raise NotImplementedError

    def validate_output(self, wav: np.ndarray, text: str) -> list[str]:
        """Cheap signal checks every engine shares; adapters may extend."""
        problems = []
        if wav.ndim != 1:
            problems.append(f"expected mono audio, got shape {wav.shape}")
        if wav.size < int(0.1 * self.sample_rate):
            problems.append("engine returned (near-)silence")
        elif float(np.max(np.abs(wav))) < 1e-4:
            problems.append("engine returned a silent clip")
        if not np.all(np.isfinite(wav)):
            problems.append("audio contains NaN/Inf samples")
        return problems

    def unload(self) -> None:
        pass

    # backwards-compatible alias used by the v2 pipeline/tests
    def close(self) -> None:
        self.unload()


def _canonical(params: dict) -> dict:
    return json.loads(json.dumps(params or {}, sort_keys=True, default=str))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def params_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]
