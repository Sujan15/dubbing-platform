"""TTS engine registry.

``engine_class(engine)`` returns the adapter class; ``load_engine(settings, device, engine)`` returns a
ready instance - in-process, or proxied into an isolated interpreter when ``<ENGINE>_PYTHON`` is set.
``engine_status(settings)`` is what the API shows: enabled / available / licence / revision pin, per
engine, without loading any model.
"""

from __future__ import annotations

from typing import Optional

from backend.app.config import Settings, TTSEngine
from backend.services.tts.base import (  # noqa: F401  (re-exported)
    BaseTTSEngine,
    EngineCapabilities,
    EngineMetadata,
    EngineUnavailable,
    TTSEngineError,
    VoiceSpec,
    params_hash,
    text_hash,
)
from backend.services.tts.indicf5 import IndicF5Engine
from backend.services.tts.parler import ParlerEngine

_REGISTRY: dict[TTSEngine, type[BaseTTSEngine]] = {
    TTSEngine.INDICF5: IndicF5Engine,
    TTSEngine.PARLER: ParlerEngine,
}


def register(engine: TTSEngine, cls: type[BaseTTSEngine]) -> None:
    """Replace an adapter (tests inject deterministic fakes this way)."""
    if engine == TTSEngine.XTTS:
        raise ValueError("XTTS has been removed and cannot be registered.")
    _REGISTRY[engine] = cls


def engine_class(engine: TTSEngine) -> type[BaseTTSEngine]:
    try:
        return _REGISTRY[TTSEngine(engine)]
    except KeyError as exc:
        raise EngineUnavailable(f"No adapter registered for engine {engine!r}") from exc


def isolated_python(settings: Settings, engine: TTSEngine) -> Optional[str]:
    return {TTSEngine.INDICF5: settings.indicf5_python, TTSEngine.PARLER: settings.parler_python}.get(engine)


def load_engine(settings: Settings, device: str, engine: TTSEngine) -> BaseTTSEngine:
    engine = TTSEngine(engine)
    if engine not in settings.enabled_tts_engines:
        raise EngineUnavailable(f"Engine '{engine.value}' is not enabled (TTS_ENGINES_ENABLED={settings.tts_engines_enabled}).")
    cls = engine_class(engine)
    python = isolated_python(settings, engine)
    if python:
        from backend.services.tts.isolated import IsolatedEngine
        return IsolatedEngine(settings, device, engine, python, cls)
    return cls(settings, device)


def engine_status(settings: Settings) -> list[dict]:
    out = []
    for engine, cls in _REGISTRY.items():
        avail = cls.check_model_availability(settings)
        caps = cls.get_capabilities()
        out.append({
            "engine": engine.value, "enabled": engine in settings.enabled_tts_engines,
            "available": bool(avail.get("available")), "reason": avail.get("reason") or "",
            "model_id": avail.get("model_id"), "model_revision": avail.get("revision"),
            "isolated_python": isolated_python(settings, engine),
            "capabilities": {"reference_conditioned": caps.reference_conditioned,
                             "needs_reference_transcript": caps.needs_reference_transcript,
                             "prompt_conditioned": caps.prompt_conditioned, "supports_seed": caps.supports_seed,
                             "supports_speed": caps.supports_speed, "languages": list(caps.languages)},
            "license": caps.license, "commercial_use": caps.commercial_use,
            "legacy": False,
        })
    return out
