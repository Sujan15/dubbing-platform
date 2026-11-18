# hindi-dubbing-platform/backend/app/config.py

"""Central configuration for the Hindi dubbing platform.

All values come from environment variables or the repository-level ``.env`` file.
Nothing here imports torch or any model library, so the API process starts fast
and configuration errors surface before any model is loaded.
"""

from __future__ import annotations

import os
import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET_LANGUAGE = "hi"
SUPPORTED_SOURCE_LANGUAGES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese (Mandarin)",
    "ko": "Korean",
}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".m4v", ".webm", ".avi"}
ALLOWED_REFERENCE_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
SUPPORTED_TARGET_LANGUAGES: dict[str, str] = {"hi": "Hindi"}
REFERENCE_SAMPLE_RATES = {16000, 22050, 24000}


class MixMode(str, Enum):
    PRESERVE_BACKGROUND = "preserve_background"
    DUB_ONLY = "dub_only"
    ORIGINAL_PLUS_DUB = "original_plus_dub"


class TranslationEngine(str, Enum):
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


class TTSEngine(str, Enum):
    """Engine identifiers persisted in records; XTTS remains only for historical record parsing."""
    XTTS = "xtts"
    INDICF5 = "indicf5"
    PARLER = "parler"


class SeparationFallback(str, Enum):
    """What generation does when no separated stems exist but ``preserve_background`` was requested.

    * ``original_plus_dub`` - explicit, reported fallback: original ducked under Hindi (v2 behaviour)
    * ``dub_only``          - drop the original soundtrack entirely (no leakage, no music)
    * ``fail``              - stop the job; the operator must enable separation or pick another mode
    """
    ORIGINAL_PLUS_DUB = "original_plus_dub"
    DUB_ONLY = "dub_only"
    FAIL = "fail"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    # ---- Server -----------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    max_upload_mb: int = Field(8192, ge=1)
    max_concurrent_jobs: int = Field(1, ge=1, le=4)

    # ---- Local Windows directories ---------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    ffmpeg_timeout_seconds: int = Field(6 * 3600, ge=60)

    # ---- Compute ----------------------------------------------------------------
    device: Literal["auto", "cuda", "cpu"] = "auto"

    # ---- Stage 2: source separation --------------------------------------------
    separation_enabled: bool = True
    separation_model: str = "htdemucs"
    separation_allow_cpu: bool = False
    separation_shifts: int = Field(1, ge=0, le=10)
    separation_block_seconds: float = Field(300.0, ge=30.0)

    # ---- Stage 3: ASR + diarization --------------------------------------------
    whisper_model: str = "large-v3"
    whisper_compute_type: str = "auto"
    whisper_beam_size: int = Field(5, ge=1, le=10)
    hf_token: Optional[SecretStr] = None
    diarization_model: str = "pyannote/speaker-diarization-3.1"

    # ---- Stage 4: translation ---------------------------------------------------
    translation_engine: TranslationEngine = TranslationEngine.OPENROUTER
    translation_fallback_engine: Optional[TranslationEngine] = None
    openrouter_api_key: Optional[SecretStr] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_num_ctx: int = Field(8192, ge=2048)
    translation_batch_size: int = Field(15, ge=1, le=60)
    translation_context_lines: int = Field(6, ge=0, le=30)
    translation_temperature: float = Field(0.3, ge=0.0, le=1.5)
    llm_timeout_seconds: float = Field(180.0, ge=10.0)
    llm_max_retries: int = Field(4, ge=1, le=10)

    # ---- Stage 6: TTS -----------------------------------------------------------
    speaker_reference_seconds: float = Field(10.0, ge=3.0, le=30.0)

    # ---- Projects, human review and the voice library --------------------------
    project_review_enabled: bool = True
    default_voices_dir: Optional[Path] = None
    min_speaker_reference_seconds: float = Field(6.0, ge=1.0, le=60.0)
    speaker_reference_sample_rate: int = 22050
    max_reference_upload_mb: int = Field(50, ge=1, le=500)

    # ---- Hybrid voice library / engines (v3) ------------------------------------
    tts_engines_enabled: str = "parler,indicf5"             # Parler primary; IndicF5 secondary
    indicf5_model: str = "ai4bharat/IndicF5"
    indicf5_revision: Optional[str] = None                 # pin a commit hash once validated
    indicf5_python: Optional[str] = None                   # interpreter of an isolated venv (optional)
    parler_model: str = "ai4bharat/indic-parler-tts"
    parler_revision: Optional[str] = None
    parler_python: Optional[str] = None
    tts_isolated_worker_timeout: int = Field(900, ge=60)
    voice_library_dir: Optional[Path] = None

    # ---- Speaker identity across episodes (v3) ----------------------------------
    speaker_identity_enabled: bool = True
    speaker_embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    min_segment_identity_duration: float = Field(1.0, ge=0.2, le=10.0)
    speaker_match_high_threshold: float = Field(0.75, ge=0.0, le=1.0)
    speaker_match_review_threshold: float = Field(0.55, ge=0.0, le=1.0)
    speaker_overlap_review: bool = True

    # ---- Separation policy + leakage (v3) --------------------------------------
    separation_fallback: SeparationFallback = SeparationFallback.ORIGINAL_PLUS_DUB
    leakage_review_threshold: float = Field(0.35, ge=0.0, le=1.0)
    leakage_fail_threshold: float = Field(0.65, ge=0.0, le=1.0)

    # ---- Professional mixing (v3) -----------------------------------------------
    dynamic_ducking_enabled: bool = True
    music_duck_db: float = Field(-6.0, le=0.0, ge=-24.0)
    duck_attack_ms: float = Field(30.0, ge=1.0, le=500.0)
    duck_release_ms: float = Field(400.0, ge=10.0, le=5000.0)
    dialogue_target_lufs: float = Field(-17.0, le=-6.0, ge=-40.0)
    dialogue_normalize_max_db: float = Field(6.0, ge=0.0, le=24.0)
    min_dialogue_bed_margin_db: float = Field(6.0, ge=0.0, le=30.0)
    true_peak_ceiling_dbtp: float = Field(-1.0, le=0.0, ge=-6.0)

    # ---- Stage 7: timing --------------------------------------------------------
    timing_atempo_threshold: float = Field(0.15, ge=0.0, le=0.5)
    timing_max_refine_attempts: int = Field(2, ge=0, le=5)
    timing_max_atempo: float = Field(1.30, ge=1.0, le=2.0,
                                     validation_alias=AliasChoices("timing_max_atempo", "timing_max_speedup"))
    timing_max_tts_speed: float = Field(1.0, ge=1.0, le=1.5)
    timing_max_overflow_seconds: float = Field(0.0, ge=0.0, le=3.0)
    loudness_match_max_db: float = Field(9.0, ge=0.0, le=24.0)

    # ---- Stage 8: mixing / muxing ----------------------------------------------
    default_mix_mode: MixMode = MixMode.PRESERVE_BACKGROUND
    duck_db: float = Field(-18.0, le=0.0)
    mix_sample_rate: int = 44100
    output_audio_bitrate: str = "192k"

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        if self.translation_fallback_engine == self.translation_engine:
            self.translation_fallback_engine = None
        self.default_voices_dir = Path(self.default_voices_dir or self.data_dir / "default_voices").expanduser().resolve()
        if self.speaker_reference_sample_rate not in REFERENCE_SAMPLE_RATES:
            raise ValueError(f"SPEAKER_REFERENCE_SAMPLE_RATE must be one of {sorted(REFERENCE_SAMPLE_RATES)}")
        self.voice_library_dir = Path(self.voice_library_dir or self.data_dir / "voice_library").expanduser().resolve()
        engines = {e.strip().lower() for e in self.tts_engines_enabled.split(",") if e.strip()}
        supported_engines = {TTSEngine.PARLER.value, TTSEngine.INDICF5.value}
        retired_engines = engines & {TTSEngine.XTTS.value}
        engines -= retired_engines
        if retired_engines and not engines:
            raise ValueError("XTTS has been removed; explicitly enable parler and/or indicf5.")
        if retired_engines:
            logging.getLogger("dubbing.config").warning("Ignoring retired XTTS in TTS_ENGINES_ENABLED.")
        unknown = engines - supported_engines
        if unknown:
            raise ValueError(f"TTS_ENGINES_ENABLED contains unsupported engines: {sorted(unknown)}")
        if not engines:
            engines = {TTSEngine.PARLER.value, TTSEngine.INDICF5.value}
        self.tts_engines_enabled = ",".join(sorted(engines))
        if self.speaker_match_review_threshold > self.speaker_match_high_threshold:
            raise ValueError("SPEAKER_MATCH_REVIEW_THRESHOLD must not exceed SPEAKER_MATCH_HIGH_THRESHOLD")
        if self.leakage_review_threshold > self.leakage_fail_threshold:
            raise ValueError("LEAKAGE_REVIEW_THRESHOLD must not exceed LEAKAGE_FAIL_THRESHOLD")
        return self

    # ---- Derived paths ----------------------------------------------------------
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def enabled_tts_engines(self) -> set[TTSEngine]:
        return {TTSEngine(e) for e in self.tts_engines_enabled.split(",") if e}

    def ensure_dirs(self) -> None:
        for d in (self.uploads_dir, self.work_dir, self.outputs_dir, self.jobs_dir, self.models_dir,
                  self.projects_dir, self.default_voices_dir, self.voice_library_dir):
            d.mkdir(parents=True, exist_ok=True)

    def apply_runtime_environment(self) -> None:
        """Process-wide environment required before any model library is imported.

        Model caches live under ``data/models`` instead of the Windows user profile.
        ``setdefault`` keeps any value the operator already exported.
        """
        models = os.fspath(self.models_dir)
        os.environ.setdefault("HF_HOME", os.path.join(models, "huggingface"))
        os.environ.setdefault("TORCH_HOME", os.path.join(models, "torch"))
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        # torch and ctranslate2 both ship libiomp5md.dll on Windows; without this the
        # process aborts with "OMP: Error #15" the first time both are loaded.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    def preflight_errors(self, translation_engine: TranslationEngine) -> list[str]:
        """Configuration problems that would make a job fail later; checked at submit time."""
        errors: list[str] = []
        if not self.hf_token or not self.hf_token.get_secret_value().strip():
            errors.append(
                "HF_TOKEN is not set. Speaker diarization needs a Hugging Face token with the terms of "
                "'pyannote/speaker-diarization-3.1' and 'pyannote/segmentation-3.0' accepted."
            )
        engines = {translation_engine}
        if self.translation_fallback_engine:
            engines.add(self.translation_fallback_engine)
        if TranslationEngine.OPENROUTER in engines and (
            not self.openrouter_api_key or not self.openrouter_api_key.get_secret_value().strip()
        ):
            errors.append("OPENROUTER_API_KEY is not set but OpenRouter is a configured translation engine.")
        return errors


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
