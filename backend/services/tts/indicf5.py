# hindi-dubbing-platform/backend/services/tts/indicf5.py
"""IndicF5 (AI4Bharat) adapter: reference-conditioned Hindi synthesis for CLONE voices.

IndicF5 conditions on a short reference recording *and its transcript*; both are part of the
``VoiceSpec`` and therefore of the TTS cache key, so re-using the same approved master reference
for a character always re-uses the same cached clips.

The model is loaded lazily (``transformers`` ``trust_remote_code`` entry point published by
AI4Bharat). Pin ``INDICF5_REVISION`` to the commit you validated; an unpinned revision is reported
in the engine status so an unreviewed upstream change can never be mistaken for the validated one.
Optional dependencies (``requirements-engines.txt``) are not installed by default: when they are
missing the engine reports ``available=false`` instead of breaking the server.
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

_OUTPUT_SAMPLE_RATE = 24000


def _install_checkpointless_f5_loader():
    """Build the remote CFM before Transformers loads the repository's safetensors."""
    from f5_tts.infer import utils_infer

    original_load_model = utils_infer.load_model

    def load_model(model_cls, model_cfg, ckpt_path=None, mel_spec_type=utils_infer.mel_spec_type,
                   vocab_file="", ode_method=utils_infer.ode_method, use_ema=True, device=utils_infer.device):
        if ckpt_path is not None:
            return original_load_model(
                model_cls, model_cfg, ckpt_path, mel_spec_type=mel_spec_type, vocab_file=vocab_file,
                ode_method=ode_method, use_ema=use_ema, device=device)
        if not vocab_file:
            raise ValueError("IndicF5 did not provide its vocabulary path")
        vocab_char_map, vocab_size = utils_infer.get_tokenizer(vocab_file, "custom")
        return utils_infer.CFM(
            transformer=model_cls(**model_cfg, text_num_embeds=vocab_size,
                                  mel_dim=utils_infer.n_mel_channels),
            mel_spec_kwargs={
                "n_fft": utils_infer.n_fft,
                "hop_length": utils_infer.hop_length,
                "win_length": utils_infer.win_length,
                "n_mel_channels": utils_infer.n_mel_channels,
                "target_sample_rate": utils_infer.target_sample_rate,
                "mel_spec_type": mel_spec_type,
            },
            odeint_kwargs={"method": ode_method},
            vocab_char_map=vocab_char_map,
        ).to(device)

    utils_infer.load_model = load_model
    return utils_infer, original_load_model


class IndicF5Engine(BaseTTSEngine):
    engine = TTSEngine.INDICF5
    sample_rate = _OUTPUT_SAMPLE_RATE
    capabilities = EngineCapabilities(
        engine=TTSEngine.INDICF5, reference_conditioned=True, needs_reference_transcript=True,
        prompt_conditioned=False, supports_seed=True, supports_speed=False, languages=("hi",),
        commercial_use=None, license="See the model card of INDICF5_MODEL (verify before commercial use)")

    @classmethod
    def validate_configuration(cls, settings: Settings) -> list[str]:
        problems = []
        if not settings.indicf5_model.strip():
            problems.append("INDICF5_MODEL is empty")
        if settings.indicf5_python and not Path(settings.indicf5_python).is_file():
            problems.append(f"INDICF5_PYTHON does not exist: {settings.indicf5_python}")
        return problems

    @classmethod
    def check_model_availability(cls, settings: Settings) -> dict:
        problems = cls.validate_configuration(settings)
        if not settings.indicf5_python and importlib.util.find_spec("transformers") is None:
            problems.append("package 'transformers' is not installed")
        note = "" if settings.indicf5_revision else "INDICF5_REVISION is not pinned"
        return {"available": not problems, "reason": "; ".join(problems) or note, "model_id": settings.indicf5_model,
                "revision": settings.indicf5_revision}

    def __init__(self, settings: Settings, device: str):
        super().__init__(settings, device)
        problems = self.validate_configuration(settings)
        if problems:
            raise EngineUnavailable(problems[0])
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise EngineUnavailable(f"IndicF5 dependencies missing ({exc}); install requirements-engines.txt "
                                    "or point INDICF5_PYTHON at an isolated venv.") from exc
        self._torch = torch
        kwargs = {"trust_remote_code": True, "cache_dir": os.fspath(settings.models_dir / "huggingface")}
        token = settings.hf_token.get_secret_value().strip() if settings.hf_token else ""
        token = token or os.environ.get("HF_TOKEN", "").strip() or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN", "").strip()
        if token:
            kwargs["token"] = token
        if settings.indicf5_revision:
            kwargs["revision"] = settings.indicf5_revision
        previous_token = os.environ.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
        loader_module = None
        original_load_model = None
        try:
            loader_module, original_load_model = _install_checkpointless_f5_loader()
            self.model = AutoModel.from_pretrained(settings.indicf5_model, **kwargs)
        finally:
            if loader_module is not None and original_load_model is not None:
                loader_module.load_model = original_load_model
            if token:
                if previous_token is None:
                    os.environ.pop("HF_TOKEN", None)
                else:
                    os.environ["HF_TOKEN"] = previous_token
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
        self.fingerprint = {"model": settings.indicf5_model, "revision": settings.indicf5_revision, "sr": self.sample_rate}

    def get_model_metadata(self) -> EngineMetadata:
        return EngineMetadata(self.engine, self.settings.indicf5_model, self.settings.indicf5_revision,
                              self.sample_rate, self.device)

    def synthesize(self, text: str, spec: VoiceSpec, seed: int, speed: float = 1.0) -> np.ndarray:
        if spec.reference_path is None or not (spec.reference_transcript or "").strip():
            raise TTSEngineError("IndicF5 needs a reference recording and its transcript.")
        self._torch.manual_seed(seed)
        with self._torch.inference_mode():
            audio = self.model(text, ref_audio_path=os.fspath(Path(spec.reference_path)),
                               ref_text=spec.reference_transcript.strip())
        wav = np.asarray(audio)
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        wav = wav.astype(np.float32).reshape(-1) if wav.ndim == 1 else wav.astype(np.float32).mean(axis=-1).reshape(-1)
        return wav

    def unload(self) -> None:
        self.model = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
