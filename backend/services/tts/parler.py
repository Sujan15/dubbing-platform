# hindi-dubbing-platform/backend/services/tts/parler.py
"""Parler-TTS adapter (AI4Bharat Indic Parler-TTS): prompt-conditioned Hindi *cast* voices.

Parler is NOT a speaker-cloning engine: the voice is whatever the saved description prompt and
the exact checkpoint produce. A cast voice therefore stores the prompt verbatim, the model id and
revision, the generation parameters and (when supported) a fixed seed; all of them are part of the
TTS cache key. A seed improves repeatability but does not guarantee identical identity - the voice
library's audition/approval workflow is the only evidence that a preset is usable in production.

``parler_tts`` pins a different ``transformers`` version from the main speech stack. Run it in an isolated venv via
``PARLER_PYTHON`` (see ``backend.services.tts.isolated``) instead of fighting the pins in the main
environment.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

from backend.app.config import Settings, TTSEngine
from backend.services.tts.base import (
    BaseTTSEngine,
    EngineCapabilities,
    EngineMetadata,
    EngineUnavailable,
    TTSEngineError,
    VoiceSpec,
)

DEFAULT_GENERATION_PARAMETERS = {"do_sample": True, "temperature": 1.0, "max_new_tokens": 2580}


class ParlerEngine(BaseTTSEngine):
    engine = TTSEngine.PARLER
    sample_rate = 44100
    capabilities = EngineCapabilities(
        engine=TTSEngine.PARLER, reference_conditioned=False, needs_reference_transcript=False,
        prompt_conditioned=True, supports_seed=True, supports_speed=False, languages=("hi",),
        commercial_use=None, license="See the model card of PARLER_MODEL (Apache-2.0 for indic-parler-tts weights; verify)")

    @classmethod
    def validate_configuration(cls, settings: Settings) -> list[str]:
        problems = []
        if not settings.parler_model.strip():
            problems.append("PARLER_MODEL is empty")
        if settings.parler_python and not Path(settings.parler_python).is_file():
            problems.append(f"PARLER_PYTHON does not exist: {settings.parler_python}")
        return problems

    @classmethod
    def check_model_availability(cls, settings: Settings) -> dict:
        problems = cls.validate_configuration(settings)
        if not settings.parler_python and importlib.util.find_spec("parler_tts") is None:
            problems.append("package 'parler_tts' is not installed in this venv (set PARLER_PYTHON to an isolated venv)")
        note = "" if settings.parler_revision else "PARLER_REVISION is not pinned"
        return {"available": not problems, "reason": "; ".join(problems) or note, "model_id": settings.parler_model,
                "revision": settings.parler_revision}

    def __init__(self, settings: Settings, device: str):
        super().__init__(settings, device)
        problems = self.validate_configuration(settings)
        if problems:
            raise EngineUnavailable(problems[0])
        try:
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise EngineUnavailable(f"Parler dependencies missing ({exc}); install requirements-engines.txt in an "
                                    "isolated venv and set PARLER_PYTHON.") from exc
        self._torch = torch
        kwargs = {
            "cache_dir": os.fspath(settings.models_dir / "huggingface"),
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float32,
        }
        token = settings.hf_token.get_secret_value().strip() if settings.hf_token else ""
        if token:
            kwargs["token"] = token
        if settings.parler_revision:
            kwargs["revision"] = settings.parler_revision
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(settings.parler_model, **kwargs).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.parler_model, **kwargs)
        description_kwargs = {"cache_dir": os.fspath(settings.models_dir / "huggingface")}
        if token:
            description_kwargs["token"] = token
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path, **description_kwargs)
        self.sample_rate = int(self.model.config.sampling_rate)
        self.fingerprint = {"model": settings.parler_model, "revision": settings.parler_revision, "sr": self.sample_rate}

    def get_model_metadata(self) -> EngineMetadata:
        return EngineMetadata(self.engine, self.settings.parler_model, self.settings.parler_revision,
                              self.sample_rate, self.device)

    def synthesize(self, text: str, spec: VoiceSpec, seed: int, speed: float = 1.0) -> np.ndarray:
        if not (spec.voice_prompt or "").strip():
            raise TTSEngineError("Parler needs the voice description prompt of the cast preset.")
        params = {**DEFAULT_GENERATION_PARAMETERS, **(spec.generation_parameters or {})}
        self._torch.manual_seed(spec.seed if spec.seed is not None else seed)
        desc = self.description_tokenizer(spec.voice_prompt.strip(), return_tensors="pt").to(self.device)
        prompt = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            generation = self.model.generate(
                input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask, **params)
        return np.asarray(generation.cpu().numpy(), dtype=np.float32).squeeze().reshape(-1)

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.description_tokenizer = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
