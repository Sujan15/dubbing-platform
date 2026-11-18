# hindi-dubbing-platform/backend/services/pipeline.py
"""DubbingPipeline: source video (English / Mandarin / Korean) -> Hindi-dubbed video.

Stages run sequentially. Each one writes into ``<work>/<video-hash>/<stage>/<key>/`` where ``key``
hashes the stage version, its parameters and the identity of its inputs. A stage directory is
built under a temporary name and renamed into place only when complete, so a crash never leaves
a half-written stage that a later run would mistake for a cache hit. Re-running the same video
with the same settings skips every completed stage; changing e.g. the translation model only
re-runs translation and everything downstream of it.

Heavy libraries (torch, faster-whisper, pyannote, demucs, TTS) are imported inside the stage
that needs them and released afterwards, so only one large model occupies GPU memory at a time.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import soundfile as sf
from pydantic import BaseModel

from backend.app.config import (
    SUPPORTED_SOURCE_LANGUAGES,
    TARGET_LANGUAGE,
    MixMode,
    SeparationFallback,
    Settings,
    TranslationEngine,
    TTSEngine,
)
from backend.services import audio_qa
from backend.services.media import FFmpeg, MediaError, MediaInfo, read_audio, rms_db, trim_silence, write_audio
from backend.services.mixing import DuckingFollower, apply_gain_db, plan_gains
from backend.services.speaker_identity import SpeakerEmbedder, attribution_confidence
from backend.services.translation import SourceLine, TranslationError, Translator, engine_fingerprint
from backend.services.tts import BaseTTSEngine, EngineUnavailable, TTSEngineError, VoiceSpec, load_engine, text_hash
from backend.services.voices import (
    CharacterLookup,
    DefaultVoiceBank,
    LibraryLookup,
    ResolvedVoice,
    VoiceLibraryLookup,
    VoiceRequest,
    VoiceResolutionError,
    VoiceSource,
    resolve_voice,
)

log = logging.getLogger("dubbing.pipeline")

ASR_SAMPLE_RATE = 16000
_COMPLETE = "_complete.json"


# =============================================================================== data model
class DialogueSegment(BaseModel):
    segment_id: str
    start_time: float
    end_time: float
    speaker_id: str
    original_text: str
    translated_text: str
    generated_audio_path: Optional[str] = None
    original_duration: float
    generated_duration: Optional[float] = None
    speed_ratio: float = 1.0
    # Optional additions (project/review jobs); absent values keep old consumers working.
    speaker_display_name: Optional[str] = None
    voice_source: Optional[str] = None


@dataclass
class JobOptions:
    source_language: Optional[str] = None  # None = auto-detect
    mix_mode: MixMode = MixMode.PRESERVE_BACKGROUND
    translation_engine: TranslationEngine = TranslationEngine.OPENROUTER
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


@dataclass
class PipelineResult:
    output_video: Path
    manifest: Path
    timing_report: Path
    source_language: str
    speakers: int
    segments: int
    warnings: list[str] = field(default_factory=list)
    voice_resolution: Optional[Path] = None
    # v3
    qa_report: Optional[Path] = None
    qa_status: Optional[str] = None
    generation_runs: Optional[Path] = None
    resolved: dict[str, ResolvedVoice] = field(default_factory=dict)


@dataclass
class MediaArtifacts:
    video: Path
    info: MediaInfo
    ws: Path
    audio_16k: Path
    master: Path
    vocals: Path
    background: Optional[Path]
    speech_16k: Path


@dataclass
class AnalysisResult:
    media: MediaArtifacts
    source_language: str
    language_probability: float
    transcript: dict
    turns: list[dict]
    blocks: list[dict]
    translations: dict[str, str]
    voices_dir: Path
    profiles: dict
    warnings: list[str] = field(default_factory=list)
    embeddings_path: Optional[Path] = None      # v3: per-speaker ECAPA embeddings (identity stage)

    def artifact_index(self) -> dict:
        """Internal artifact locations, persisted per job so review/generation never re-run analysis."""
        m = self.media
        return {
            "workspace": os.fspath(m.ws), "master": os.fspath(m.master), "vocals": os.fspath(m.vocals),
            "background": os.fspath(m.background) if m.background else None,
            "voices_dir": os.fspath(self.voices_dir), "duration": m.info.duration,
            "source_language": self.source_language, "language_probability": self.language_probability,
            "turns": self.turns, "warnings": self.warnings,
            "embeddings": os.fspath(self.embeddings_path) if self.embeddings_path else None,
            "separation": m.background is not None,
        }


@dataclass
class GenerationPlan:
    """What to synthesize: approved (or automatic) dialogue lines and per-speaker voice requests."""
    blocks: list[dict]                       # {id, start, end, speaker, text(original)}
    translations: dict[str, str]             # id -> Hindi line to speak
    source_language: str
    voice_requests: dict[str, VoiceRequest] = field(default_factory=dict)
    locked: set[str] = field(default_factory=set)  # human-edited lines the LLM must not rewrite
    library: Optional[LibraryLookup] = None
    voice_library: Optional[VoiceLibraryLookup] = None    # v3: approved Voice Library voices
    characters: Optional[CharacterLookup] = None          # v3: canonical character -> active voice profile


class PipelineError(RuntimeError):
    pass


ProgressFn = Callable[[str, float, str], None]  # (stage, overall fraction 0..1, message)

ANALYSIS_STAGES: list[tuple[str, float]] = [
    ("validate", 1), ("extract", 3), ("separate", 14), ("transcribe", 18), ("diarize", 10),
    ("align", 1), ("translate", 10), ("voices", 2), ("identity", 2),
]
GENERATION_STAGES: list[tuple[str, float]] = [
    ("prepare", 2), ("voices", 2), ("resolve_voices", 1), ("synthesize", 73), ("mix", 10), ("mux", 8),
    ("validate_output", 2), ("qa", 2),
]
_STAGES: list[tuple[str, float]] = ANALYSIS_STAGES + [
    ("resolve_voices", 1), ("synthesize", 33), ("mix", 5), ("mux", 3), ("validate_output", 1), ("qa", 1),
]
_PREPARE_ALIAS = {"validate": "prepare", "extract": "prepare", "separate": "prepare"}


class _Progress:
    def __init__(self, fn: ProgressFn, stages: Optional[list[tuple[str, float]]] = None,
                 alias: Optional[dict[str, str]] = None):
        self.fn = fn
        self.last = 0.0
        self.alias = alias or {}
        stages = stages or _STAGES
        total = sum(w for _, w in stages)
        self.span: dict[str, tuple[float, float]] = {}
        acc = 0.0
        for name, w in stages:
            self.span.setdefault(name, (acc / total, w / total))
            acc += w

    def __call__(self, stage: str, fraction: float = 0.0, message: str = "") -> None:
        stage = self.alias.get(stage, stage)
        if stage in self.span:
            start, width = self.span[stage]
            value = round(min(1.0, start + width * max(0.0, min(1.0, fraction))), 4)
            self.last = max(self.last, value)  # never report progress going backwards
        self.fn(stage, self.last, message)


# =============================================================================== pipeline
class DubbingPipeline:
    def __init__(self, settings: Settings):
        self.s = settings
        self.ff = FFmpeg(settings.ffmpeg_bin, settings.ffprobe_bin, settings.ffmpeg_timeout_seconds)
        self.bank = DefaultVoiceBank(settings)
        self._digests: dict[tuple, str] = {}
        self._lock = threading.Lock()
        self._device: Optional[str] = None

    # ------------------------------------------------------------------ entry points
    def run(self, video: Path, options: JobOptions, job_output_dir: Path, progress: ProgressFn) -> PipelineResult:
        """Single-shot (non-project) job: analysis and generation with automatic voice resolution."""
        p = _Progress(progress, _STAGES)
        analysis = self.analyze(video, options, p)
        requests = {
            speaker_id: VoiceRequest(
                speaker_id=speaker_id,
                display_name=speaker_id,
                voice_source=VoiceSource.CAST,
                voice_prompt="A natural Hindi speaker with clear diction and a moderate conversational pace.",
            )
            for speaker_id in sorted({block["speaker"] for block in analysis.blocks})
        }
        plan = GenerationPlan(blocks=analysis.blocks, translations=analysis.translations,
                              source_language=analysis.source_language, voice_requests=requests)
        return self.generate(analysis.media, analysis.turns, plan, options, job_output_dir, p,
                             list(analysis.warnings))

    def analyze_for_review(self, video: Path, options: JobOptions, job_output_dir: Path,
                           progress: ProgressFn) -> AnalysisResult:
        """Project job, phase 1: everything up to translation + clone references. No TTS."""
        job_output_dir.mkdir(parents=True, exist_ok=True)
        analysis = self.analyze(video, options, _Progress(progress, ANALYSIS_STAGES))
        _write_json(job_output_dir / "analysis.json", analysis.artifact_index())
        return analysis

    def generate_reviewed(self, video: Path, turns: list[dict], plan: GenerationPlan, options: JobOptions,
                          job_output_dir: Path, progress: ProgressFn) -> PipelineResult:
        """Project job, phase 2: approved draft -> voices -> TTS -> timing -> mix -> mux. Upstream stages
        are cache hits (probe + hash only); ASR, diarization and translation are never re-run here."""
        p = _Progress(progress, GENERATION_STAGES, alias=_PREPARE_ALIAS)
        warnings: list[str] = []
        media = self._media(Path(video), p, warnings)
        return self.generate(media, turns, plan, options, job_output_dir, p, warnings)

    # ------------------------------------------------------------------ analysis
    def _media(self, video: Path, p: _Progress, warnings: list[str]) -> MediaArtifacts:
        video = Path(video).resolve()
        p("validate", 0.0, "Probing input video")
        info = self.ff.probe(video)
        log.info("Input %s: %.1fs, %sx%s @ %.3f fps, video=%s, audio streams=%d",
                 video.name, info.duration, info.width, info.height, info.fps, info.video_codec, info.audio_streams)
        ws = self.s.work_dir / self._digest(video)[:16]
        ws.mkdir(parents=True, exist_ok=True)
        log.info("Workspace: %s | device: %s", ws, self.device)
        p("validate", 1.0, "Input validated")

        p("extract", 0.0, "Extracting audio")
        d_extract = self._stage(ws, "extract", 1, {"asr_sr": ASR_SAMPLE_RATE, "mix_sr": self.s.mix_sample_rate},
                                [video], lambda out: self._extract(video, out))
        audio_16k, master = d_extract / "audio_16k.wav", d_extract / "master.wav"
        p("extract", 1.0, "Audio extracted")

        d_sep = self._maybe_separate(ws, master, p, warnings)
        p("separate", 1.0, "Separation done" if d_sep else "Separation skipped")
        return MediaArtifacts(
            video=video, info=info, ws=ws, audio_16k=audio_16k, master=master,
            vocals=d_sep / "vocals.wav" if d_sep else master,
            background=d_sep / "background.wav" if d_sep else None,
            speech_16k=d_sep / "vocals_16k.wav" if d_sep else audio_16k)

    def analyze(self, video: Path, options: JobOptions, p: _Progress) -> AnalysisResult:
        warnings: list[str] = []
        media = self._media(Path(video), p, warnings)
        speech_16k, info = media.speech_16k, media.info
        ws = media.ws

        # ASR ---------------------------------------------------------------------
        asr_params = {"model": self.s.whisper_model, "compute": self._whisper_compute_type(),
                      "beam": self.s.whisper_beam_size, "language": options.source_language}
        d_asr = self._stage(ws, "transcribe", 1, asr_params, [speech_16k],
                            lambda out: self._transcribe(speech_16k, out, options.source_language, info.duration, p))
        transcript = _read_json(d_asr / "transcript.json")
        lang = transcript["language"]
        if lang not in SUPPORTED_SOURCE_LANGUAGES:
            msg = f"Detected source language '{lang}' is outside the tuned set (en, zh, ko); continuing."
            log.warning(msg)
            warnings.append(msg)
        if not transcript["segments"]:
            raise PipelineError("No speech was recognised in the input; nothing to dub.")
        p("transcribe", 1.0, f"{len(transcript['segments'])} ASR segments ({lang})")

        # diarization -------------------------------------------------------------
        dia_params = {"model": self.s.diarization_model, "min": options.min_speakers, "max": options.max_speakers}
        d_dia = self._stage(ws, "diarize", 1, dia_params, [speech_16k],
                            lambda out: self._diarize(speech_16k, out, options, p))
        turns = _read_json(d_dia / "diarization.json")
        p("diarize", 1.0, f"{len({t['speaker'] for t in turns})} speakers")

        # speaker-attributed dialogue blocks -------------------------------------
        d_align = self._stage(ws, "align", 1, {}, [d_asr / "transcript.json", d_dia / "diarization.json"],
                              lambda out: _write_json(out / "blocks.json",
                                                      build_dialogue_blocks(transcript["segments"], turns)))
        blocks = _read_json(d_align / "blocks.json")
        for b in blocks:  # in-memory annotation only: blocks.json (a translation input) stays byte-identical
            b["speaker_confidence"], _flags = attribution_confidence(b, turns)
        speakers = sorted({b["speaker"] for b in blocks})
        p("align", 1.0, f"{len(blocks)} dialogue blocks, {len(speakers)} speakers")

        # translation -------------------------------------------------------------
        lang_name = SUPPORTED_SOURCE_LANGUAGES.get(lang, lang)
        tr_params = {"primary": engine_fingerprint(options.translation_engine, self.s),
                     "batch": self.s.translation_batch_size, "context": self.s.translation_context_lines}
        d_tr = self._stage(ws, "translate", 1, tr_params, [d_align / "blocks.json"],
                           lambda out: self._translate(blocks, lang_name, options, out, p))
        translations = _read_json(d_tr / "translations.json")
        p("translate", 1.0, "Translation complete")

        # clone references (cheap: audio slicing only) ----------------------------
        d_voice, profiles = self._voices_stage(ws, blocks, turns, media.vocals, p)

        # speaker identity embeddings (cross-episode suggestions) -------------------
        embeddings_path = self._identity_stage(ws, d_voice, profiles, p, warnings)
        return AnalysisResult(media=media, source_language=lang,
                              language_probability=float(transcript.get("language_probability") or 0.0),
                              transcript=transcript, turns=turns, blocks=blocks, translations=translations,
                              voices_dir=d_voice, profiles=profiles, warnings=warnings, embeddings_path=embeddings_path)

    def _identity_stage(self, ws: Path, d_voice: Path, profiles: dict, p: _Progress,
                        warnings: list[str]) -> Optional[Path]:
        if not self.s.speaker_identity_enabled:
            p("identity", 1.0, "Speaker identity disabled")
            return None
        refs = [d_voice / prof["reference"] for prof in profiles.values() if prof.get("reference")]
        if not refs:
            p("identity", 1.0, "No reference speech to embed")
            return None
        if not SpeakerEmbedder.available():
            msg = ("Speaker identity suggestions skipped: package 'speechbrain' is not installed "
                   "(pip install speechbrain to enable cross-episode character matching).")
            if msg not in warnings:
                warnings.append(msg)
            log.warning(msg)
            p("identity", 1.0, "Speaker identity unavailable")
            return None
        d = self._stage(ws, "identity", 1, {"model": self.s.speaker_embedding_model}, refs,
                        lambda out: self._embed_speakers(d_voice, profiles, out, p))
        p("identity", 1.0, "Speaker embeddings ready")
        return d / "embeddings.json"

    def _embed_speakers(self, d_voice: Path, profiles: dict, out: Path, p: _Progress) -> None:
        p("identity", 0.0, "Loading speaker embedding model")
        embedder = SpeakerEmbedder(self.s, self.device)
        vectors: dict[str, list[float]] = {}
        try:
            for i, (spk, prof) in enumerate(sorted(profiles.items())):
                if prof.get("reference"):
                    vectors[spk] = embedder.embed_file(d_voice / prof["reference"]).round(6).tolist()
                p("identity", (i + 1) / max(len(profiles), 1), f"Embedding {spk}")
        finally:
            embedder.unload()
            _free_memory()
        _write_json(out / "embeddings.json", {"model": self.s.speaker_embedding_model, "speakers": vectors})

    def _voices_stage(self, ws: Path, blocks: list[dict], turns: list[dict], vocals: Path,
                      p: _Progress) -> tuple[Path, dict]:
        geometry = {"blocks": [[b["start"], b["end"], b["speaker"]] for b in blocks],
                    "turns": [[t["start"], t["end"], t["speaker"]] for t in turns]}
        params = {"seconds": self.s.speaker_reference_seconds,
                  "geometry": hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()}
        d_voice = self._stage(ws, "voices", 2, params, [vocals],
                              lambda out: self._voice_profiles(blocks, turns, vocals, out))
        profiles = _read_json(d_voice / "profiles.json")
        p("voices", 1.0, f"{len(profiles)} voice references")
        return d_voice, profiles

    # ------------------------------------------------------------------ generation
    def generate(self, media: MediaArtifacts, turns: list[dict], plan: GenerationPlan, options: JobOptions,
                 job_output_dir: Path, p: _Progress, warnings: list[str]) -> PipelineResult:
        job_output_dir.mkdir(parents=True, exist_ok=True)
        ws, vocals, master, background, info = media.ws, media.vocals, media.master, media.background, media.info
        blocks = plan.blocks
        speakers = sorted({b["speaker"] for b in blocks})
        missing = [b["id"] for b in blocks if not (plan.translations.get(b["id"]) or "").strip()]
        if missing:
            raise PipelineError(f"No Hindi text for segment(s): {', '.join(missing[:10])}")

        # clone references for the final speaker assignment ------------------------
        d_voice, profiles = self._voices_stage(ws, blocks, turns, vocals, p)

        # voice resolution ----------------------------------------------------------
        p("resolve_voices", 0.0, "Resolving voices")
        resolved: dict[str, ResolvedVoice] = {}
        for spk in speakers:
            req = plan.voice_requests.get(spk) or VoiceRequest(speaker_id=spk, display_name=spk)
            prof = profiles.get(spk) or {}
            ref = d_voice / prof["reference"] if prof.get("reference") else None
            try:
                rv = resolve_voice(req, ref, float(prof.get("available_seconds", 0.0)), plan.library,
                                   self.bank, self.s, voice_library=plan.voice_library, characters=plan.characters,
                                   clone_transcript=prof.get("transcript"))
            except VoiceResolutionError as exc:
                raise PipelineError(str(exc)) from exc
            if rv.engine not in self.s.enabled_tts_engines:
                raise PipelineError(f"{rv.display_name}: engine '{rv.engine.value}' is not enabled "
                                    f"(TTS_ENGINES_ENABLED={self.s.tts_engines_enabled}).")
            if rv.resolved_source == VoiceSource.CLONE:
                rv.reference_seconds = float(prof.get("seconds", 0.0))
            for w in rv.warnings:
                log.warning(w)
            warnings.extend(rv.warnings)
            resolved[spk] = rv
            log.info("Voice %s (%s): requested=%s resolved=%s ref=%.1fs", spk, rv.display_name,
                     rv.requested_source.value, rv.resolved_source.value, rv.reference_seconds)
        voice_resolution = job_output_dir / "voice_resolution.json"
        _write_json(voice_resolution, {spk: rv.public() for spk, rv in resolved.items()})
        p("resolve_voices", 1.0, f"{len(resolved)} voices resolved")

        # TTS + timing alignment ----------------------------------------------------
        lang_name = SUPPORTED_SOURCE_LANGUAGES.get(plan.source_language, plan.source_language)
        specs = {spk: rv.spec for spk, rv in resolved.items()}
        voice_keys = {spk: self._spec_key(spec) for spk, spec in specs.items()}
        plan_key = {"lines": [[b["id"], b["start"], b["end"], b["speaker"], plan.translations[b["id"]],
                               b["id"] in plan.locked] for b in blocks],
                    "refs": voice_keys}
        syn_params = {"lang": TARGET_LANGUAGE, "engines": self._engine_pins(specs),
                      "translation": engine_fingerprint(options.translation_engine, self.s),
                      "plan": hashlib.sha256(json.dumps(plan_key, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                      "threshold": self.s.timing_atempo_threshold, "refine": self.s.timing_max_refine_attempts,
                      "max_atempo": self.s.timing_max_atempo, "tts_speed": self.s.timing_max_tts_speed,
                      "overflow": self.s.timing_max_overflow_seconds, "loud": self.s.loudness_match_max_db,
                      "sr": self.s.mix_sample_rate}
        meta = {spk: {"name": rv.display_name, "source": rv.resolved_source.value, "engine": rv.engine.value,
                      "voice_id": rv.voice_id, "character_id": rv.character_id, "voice_profile_id": rv.voice_profile_id}
                for spk, rv in resolved.items()}
        d_syn = self._stage(ws, "synthesize", 2, syn_params, [vocals],
                            lambda out: self._synthesize(ws, blocks, plan.translations, specs, plan.locked, meta,
                                                         vocals, info.duration, lang_name, options, out, p))
        segments = [DialogueSegment(**s) for s in _read_json(d_syn / "segments.json")]
        for seg in segments:  # stage-relative clip paths -> absolute paths for consumers
            seg.generated_audio_path = os.fspath(d_syn / seg.generated_audio_path)
        p("synthesize", 1.0, f"{len(segments)} Hindi clips aligned")

        # mixing --------------------------------------------------------------------
        mode = options.mix_mode
        separation_fallback = False
        if mode == MixMode.PRESERVE_BACKGROUND and background is None:
            separation_fallback = True
            policy = self.s.separation_fallback
            if policy == SeparationFallback.FAIL:
                raise PipelineError("preserve_background needs separated stems and SEPARATION_FALLBACK=fail: enable "
                                    "separation (SEPARATION_ALLOW_CPU=true or a CUDA GPU) or choose another mix mode.")
            mode = MixMode(policy.value)
            msg = (f"preserve_background needs separated stems; falling back to {mode.value} "
                   f"(SEPARATION_FALLBACK). The original dialogue was NOT removed"
                   + (f"; it is ducked {self.s.duck_db:.0f} dB under the Hindi lines." if mode == MixMode.ORIGINAL_PLUS_DUB
                      else "; the original soundtrack (music/SFX included) is dropped."))
            log.warning(msg)
            warnings.append(msg)
        mix_params = {"mode": mode.value, "duck": self.s.duck_db, "dynamic": self.s.dynamic_ducking_enabled,
                      "music_duck": self.s.music_duck_db, "attack": self.s.duck_attack_ms, "release": self.s.duck_release_ms,
                      "target": self.s.dialogue_target_lufs, "norm_max": self.s.dialogue_normalize_max_db,
                      "margin": self.s.min_dialogue_bed_margin_db}
        d_mix = self._stage(ws, "mix", 2, mix_params,
                            [d_syn / "segments.json", master] + ([background] if background else []),
                            lambda out: self._mix(segments, mode, master, background, out, p))
        p("mix", 1.0, f"Mixed ({mode.value})")

        # mux + output validation --------------------------------------------------
        video = media.video
        p("mux", 0.0, "Muxing Hindi audio into video")
        output = job_output_dir / f"{_safe_stem(video.stem)}.hindi{video.suffix.lower()}"
        self.ff.mux(video, d_mix / "mix.wav", output, self.s.output_audio_bitrate)
        p("mux", 1.0, "Muxed")
        p("validate_output", 0.0, "Validating output")
        self._validate_output(output, info)

        manifest = job_output_dir / "manifest.json"
        _write_json(manifest, [s.model_dump() for s in segments])
        timing_report = job_output_dir / "timing_report.json"
        shutil.copyfile(d_syn / "timing_report.json", timing_report)
        generation_runs = job_output_dir / "generation_runs.json"
        if (d_syn / "generation_runs.json").is_file():
            shutil.copyfile(d_syn / "generation_runs.json", generation_runs)
        p("validate_output", 1.0, "Output validated")

        # quality gate ----------------------------------------------------------------
        p("qa", 0.0, "Measuring quality")
        mix_report = _read_json(d_mix / "mix_report.json") if (d_mix / "mix_report.json").is_file() else {}
        voices_public = {spk: rv.public() for spk, rv in resolved.items()}
        try:
            qa = audio_qa.build_qa_report(
                settings=self.s, ff=self.ff, mode_used=mode, mode_requested=options.mix_mode,
                segments=[s.model_dump() for s in segments], timing=_read_json(timing_report), mix_dir=d_mix,
                master=master, vocals=vocals, background=background, output=output,
                separation_fallback=separation_fallback, voice_resolution=voices_public, mix_report=mix_report)
        except (MediaError, OSError, ValueError) as exc:  # QA must never turn a finished dub into a failed job
            log.error("QA report failed: %s", exc)
            qa = {"status": "REVIEW_REQUIRED", "checks": {"qa_error": {"status": "REVIEW_REQUIRED", "detail": str(exc)[:300]}},
                  "metrics": {}, "release_blocked": False}
        qa["stems"] = {"dialogue": os.fspath(d_mix / "dialogue.wav"), "bed": os.fspath(d_mix / "bed.wav"),
                       "mix": os.fspath(d_mix / "mix.wav"), "master": os.fspath(master), "vocals": os.fspath(vocals),
                       "background": os.fspath(background) if background else None}
        qa_report = job_output_dir / "qa_report.json"
        audio_qa.write_report(qa_report, qa)
        for name, check in qa.get("checks", {}).items():
            if check.get("status") in ("REVIEW_REQUIRED", "FAILED"):
                warnings.append(f"QA {check['status']}: {name} - {check.get('detail', '')}")
        p("qa", 1.0, f"QA {qa['status']}")

        _write_json(job_output_dir / "job_report.json", {
            "source_language": plan.source_language, "target_language": TARGET_LANGUAGE,
            "mix_mode_requested": options.mix_mode.value, "mix_mode_used": mode.value,
            "separation": background is not None, "separation_fallback": separation_fallback,
            "speakers": len(speakers), "segments": len(segments),
            "voices": voices_public, "engines": sorted({rv.engine.value for rv in resolved.values()}),
            "qa_status": qa["status"], "qa_metrics": qa.get("metrics", {}),
            "warnings": list(dict.fromkeys(warnings)), "runtime": self.runtime_info(),
            "output": {"file": output.name, "duration": info.duration},
        })
        return PipelineResult(output, manifest, timing_report, plan.source_language, len(speakers), len(segments),
                              list(dict.fromkeys(warnings)), voice_resolution, qa_report=qa_report,
                              qa_status=qa["status"], generation_runs=generation_runs if generation_runs.is_file() else None,
                              resolved=resolved)

    # ------------------------------------------------------------------ engines
    def _spec_key(self, spec: VoiceSpec) -> dict:
        digest = self._digest(spec.reference_path) if spec.reference_path else None
        return spec.cache_fingerprint(digest)

    def _engine_pins(self, specs: dict[str, VoiceSpec]) -> dict:
        pins = {}
        for spec in specs.values():
            if spec.engine == TTSEngine.INDICF5:
                pins["indicf5"] = [self.s.indicf5_model, self.s.indicf5_revision]
            elif spec.engine == TTSEngine.PARLER:
                pins["parler"] = [self.s.parler_model, self.s.parler_revision]
            else:
                raise PipelineError(f"TTS engine '{spec.engine.value}' has been removed.")
        return pins

    def _load_engine(self, engine: TTSEngine) -> BaseTTSEngine:
        try:
            return load_engine(self.s, self.device, engine)
        except EngineUnavailable as exc:
            raise PipelineError(f"TTS engine '{engine.value}' is unavailable: {exc}") from exc

    def runtime_info(self) -> dict:
        return {"device": self._device, "whisper": self.s.whisper_model, "whisper_compute": self.s.whisper_compute_type,
                "diarization": self.s.diarization_model,
                "tts_engines": self.s.tts_engines_enabled,
                "separation": self.s.separation_model if self.s.separation_enabled else None,
                "python": sys.version.split()[0]}

    # ------------------------------------------------------------------ device
    @property
    def device(self) -> str:
        if self._device is None:
            with self._lock:
                if self._device is None:
                    self._device = self._resolve_device()
        return self._device

    def _resolve_device(self) -> str:
        if self.s.device == "cpu":
            return "cpu"
        try:
            import torch
            cuda = bool(torch.cuda.is_available())
        except ImportError:
            cuda = False
        if self.s.device == "cuda" and not cuda:
            raise PipelineError("DEVICE=cuda but no CUDA device is available to PyTorch.")
        if cuda:
            import torch
            log.info("CUDA device: %s", torch.cuda.get_device_name(0))
            _register_torch_cuda_dlls()
            return "cuda"
        log.warning("No CUDA GPU detected; inference runs on CPU (correct but slow).")
        return "cpu"

    def _whisper_compute_type(self) -> str:
        if self.s.whisper_compute_type != "auto":
            return self.s.whisper_compute_type
        return "float16" if self.device == "cuda" else "float32"

    # ------------------------------------------------------------------ cache
    def _stage(self, ws: Path, name: str, version: int, params: dict, inputs: list[Path],
               produce: Callable[[Path], None]) -> Path:
        key_src = {"stage": name, "version": version, "params": params,
                   "inputs": [self._digest(Path(i)) for i in inputs]}
        key = hashlib.sha256(json.dumps(key_src, sort_keys=True, default=str).encode()).hexdigest()[:16]
        root = ws / name
        out = root / key
        if (out / _COMPLETE).is_file():
            log.info("[%s] cache hit (%s)", name, key)
            return out
        log.info("[%s] running (%s)", name, key)
        tmp = root / f".tmp-{key}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True)
        started = time.time()
        try:
            produce(tmp)
            _write_json(tmp / _COMPLETE, {"stage": name, "key": key, "key_source": key_src,
                                          "seconds": round(time.time() - started, 2),
                                          "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            if out.exists():
                if (out / _COMPLETE).is_file():  # a concurrent job finished the same stage first
                    shutil.rmtree(tmp, ignore_errors=True)
                    return out
                shutil.rmtree(out)
            os.replace(tmp, out)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        log.info("[%s] completed in %.1fs", name, time.time() - started)
        return out

    def _digest(self, path: Path) -> str:
        """Identity of a stage input: stage-relative for cached outputs, content SHA-256 otherwise."""
        path = path.resolve()
        for parent in list(path.parents)[:3]:
            if (parent / _COMPLETE).is_file():
                return f"stage:{parent.parent.name}/{parent.name}/{path.relative_to(parent).as_posix()}"
        st = path.stat()
        k = (os.fspath(path), st.st_size, st.st_mtime_ns)
        with self._lock:
            cached = self._digests.get(k)
        if cached:
            return cached
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        with self._lock:
            self._digests[k] = digest
        return digest

    # ------------------------------------------------------------------ stage 2
    def _extract(self, video: Path, out: Path) -> None:
        self.ff.extract_audio(video, out / "audio_16k.wav", ASR_SAMPLE_RATE, 1)       # ASR / diarization
        self.ff.extract_audio(video, out / "master.wav", self.s.mix_sample_rate, 2)    # separation / mixing

    # ------------------------------------------------------------------ stage 3
    def _maybe_separate(self, ws: Path, master: Path, p: _Progress, warnings: list[str]) -> Optional[Path]:
        if not self.s.separation_enabled:
            msg = "Source separation disabled (SEPARATION_ENABLED=false); using the full mixed track."
        elif self.device != "cuda" and not self.s.separation_allow_cpu:
            msg = ("No CUDA GPU: skipping Demucs and using the full mixed track for ASR, voice cloning and "
                   "mixing. Set SEPARATION_ALLOW_CPU=true to separate on CPU (slow, much better quality).")
        else:
            params = {"model": self.s.separation_model, "shifts": self.s.separation_shifts,
                      "block": self.s.separation_block_seconds}
            return self._stage(ws, "separate", 1, params, [master], lambda out: self._demucs(master, out, p))
        log.warning(msg)
        warnings.append(msg)
        return None

    def _demucs(self, master: Path, out: Path, p: _Progress) -> None:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        p("separate", 0.0, f"Loading {self.s.separation_model}")
        model = get_model(self.s.separation_model)
        model.eval()
        try:
            sources = list(model.sources)
            if "vocals" not in sources:
                raise PipelineError(f"Demucs model {self.s.separation_model} has no 'vocals' source.")
            vi = sources.index("vocals")
            src = master
            if model.samplerate != self.s.mix_sample_rate:
                src = self.ff.resample(master, out / "_model_input.wav", model.samplerate, model.audio_channels)
            sr = model.samplerate
            total = sf.info(os.fspath(src)).frames
            block = int(self.s.separation_block_seconds * sr)
            context = int(5.0 * sr)

            # Demucs normalises by the statistics of the whole track; compute them in a streaming pass.
            s1 = s2 = 0.0
            for start in range(0, total, block):
                mono = read_audio(src, frames=min(block, total - start), start=start)[0].mean(axis=1, dtype=np.float64)
                s1 += float(mono.sum())
                s2 += float(np.square(mono).sum())
            mean = s1 / max(total, 1)
            std = math.sqrt(max(s2 / max(total, 1) - mean ** 2, 1e-12))

            device = torch.device(self.device)
            with sf.SoundFile(os.fspath(out / "vocals.wav"), "w", sr, model.audio_channels, "FLOAT") as fv, \
                    sf.SoundFile(os.fspath(out / "background.wav"), "w", sr, model.audio_channels, "FLOAT") as fb:
                for start in range(0, total, block):
                    a, b = max(0, start - context), min(total, start + block + context)
                    data = read_audio(src, frames=b - a, start=a)[0]
                    wav = torch.from_numpy(np.ascontiguousarray(data.T))
                    wav = (wav - mean) / std
                    with torch.inference_mode():
                        est = apply_model(model, wav[None], shifts=self.s.separation_shifts, split=True,
                                          overlap=0.25, progress=False, device=device)[0]
                    est = (est * std + mean).cpu()
                    lo = start - a
                    hi = lo + min(block, total - start)
                    fv.write(est[vi, :, lo:hi].numpy().T)
                    fb.write(est[[i for i in range(len(sources)) if i != vi], :, lo:hi].sum(dim=0).numpy().T)
                    p("separate", min(1.0, (start + block) / total), "Separating vocals / background")
            (out / "_model_input.wav").unlink(missing_ok=True)
        finally:
            del model
            _free_memory()
        self.ff.resample(out / "vocals.wav", out / "vocals_16k.wav", ASR_SAMPLE_RATE, 1)

    # ------------------------------------------------------------------ stage 4
    def _transcribe(self, audio: Path, out: Path, language: Optional[str], duration: float, p: _Progress) -> None:
        _ = self.device  # registers CUDA DLL directories before ctranslate2 loads
        from faster_whisper import WhisperModel

        p("transcribe", 0.0, f"Loading Whisper {self.s.whisper_model}")
        model = WhisperModel(self.s.whisper_model, device=self.device, compute_type=self._whisper_compute_type(),
                             download_root=os.fspath(self.s.models_dir / "whisper"))
        try:
            seg_iter, info = model.transcribe(
                os.fspath(audio), language=language, task="transcribe", beam_size=self.s.whisper_beam_size,
                word_timestamps=True, vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,
            )
            log.info("Whisper language: %s (p=%.2f)", info.language, info.language_probability)
            segments = []
            for s in seg_iter:
                words = [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word,
                          "probability": round(w.probability, 4)} for w in (s.words or [])]
                segments.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
                                 "avg_logprob": round(s.avg_logprob, 4), "no_speech_prob": round(s.no_speech_prob, 4),
                                 "words": words})
                p("transcribe", s.end / max(duration, 1e-6), f"{len(segments)} segments")
        finally:
            del model
            _free_memory()
        _write_json(out / "transcript.json", {"language": info.language,
                                              "language_probability": round(info.language_probability, 4),
                                              "segments": [s for s in segments if s["text"]]})

    # ------------------------------------------------------------------ stage 5
    def _diarize(self, audio: Path, out: Path, options: JobOptions, p: _Progress) -> None:
        import torch
        from pyannote.audio import Pipeline

        token = self.s.hf_token.get_secret_value() if self.s.hf_token else None
        p("diarize", 0.0, "Loading pyannote")
        with _trusted_checkpoint_loading():
            pipe = Pipeline.from_pretrained(self.s.diarization_model, use_auth_token=token)
        if pipe is None:
            raise PipelineError(
                f"Could not load {self.s.diarization_model}. Accept the user conditions of "
                "pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0 on huggingface.co "
                "with the account that owns HF_TOKEN.")
        try:
            pipe.to(torch.device(self.device))
            data, sr = read_audio(audio)
            waveform = torch.from_numpy(np.ascontiguousarray(data.mean(axis=1, keepdims=True).T))
            kwargs = {k: v for k, v in (("min_speakers", options.min_speakers),
                                        ("max_speakers", options.max_speakers)) if v}

            def hook(step_name, step_artifact, file=None, total=None, completed=None):
                if total and completed is not None:
                    p("diarize", completed / total, f"Diarization: {step_name}")

            annotation = pipe({"waveform": waveform, "sample_rate": sr}, hook=hook, **kwargs)
            turns = [{"start": round(float(seg.start), 3), "end": round(float(seg.end), 3), "speaker": str(label)}
                     for seg, _, label in annotation.itertracks(yield_label=True)]
        finally:
            del pipe
            _free_memory()
        turns.sort(key=lambda t: t["start"])
        _write_json(out / "diarization.json", turns)

    # ------------------------------------------------------------------ stage 7
    def _translate(self, blocks: list[dict], lang_name: str, options: JobOptions, out: Path, p: _Progress) -> None:
        lines = [SourceLine(b["id"], b["speaker"], b["text"], b["end"] - b["start"]) for b in blocks]
        with Translator(self.s, options.translation_engine) as tr:
            result = tr.translate(lines, lang_name,
                                  progress=lambda f: p("translate", f, f"Translating to Hindi ({int(f * 100)}%)"))
        _write_json(out / "translations.json", result)

    # ------------------------------------------------------------------ stage 8
    def _voice_profiles(self, blocks: list[dict], turns: list[dict], vocals: Path, out: Path) -> None:
        sr = sf.info(os.fspath(vocals)).samplerate
        gap = np.zeros(int(0.2 * sr), dtype=np.float32)
        profiles = {}
        for spk in sorted({b["speaker"] for b in blocks}):
            spans = select_reference_spans(blocks, turns, spk, self.s.speaker_reference_seconds)
            available = usable_reference_seconds(blocks, turns, spk)
            pieces = []
            for s, e in spans:
                data = read_audio(vocals, frames=int((e - s) * sr), start=int(s * sr))[0]
                pieces += [data.mean(axis=1), gap]
            ref = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
            peak = float(np.max(np.abs(ref))) if ref.size else 0.0
            if peak < 1e-4:  # recorded, not raised: resolution decides (fallback voice or a clear error)
                log.warning("Speaker %s has no usable reference audio in this video.", spk)
                profiles[spk] = {"reference": None, "seconds": 0.0, "available_seconds": 0.0, "spans": []}
                continue
            ref = ref * (0.9 / peak)
            name = f"ref_{_safe_stem(spk)}.wav"
            write_audio(out / name, ref.astype(np.float32), sr, subtype="PCM_16")
            seconds = round(sum(e - s for s, e in spans), 2)
            quality: list[str] = []
            if seconds < 3.0:
                log.warning("Speaker %s has only %.1fs of reference audio; cloning will be less faithful.", spk, seconds)
                quality.append(f"only {seconds:.1f}s of clean speech")
            # transcript of the spans (source language) for reference-conditioned engines that need it; exact only
            # when every span covers a whole dialogue block
            texts, exact = [], True
            for s_, e_ in spans:
                blk = next((b for b in blocks if b["speaker"] == spk and b["start"] - 1e-3 <= s_ < b["end"]), None)
                if blk is None:
                    exact = False
                    continue
                texts.append(blk["text"])
                if e_ < blk["end"] - 0.05:
                    exact = False
            if not exact:
                quality.append("reference transcript is approximate (span cut inside a line)")
            profiles[spk] = {"reference": name, "seconds": seconds, "available_seconds": round(available, 2),
                             "spans": spans, "transcript": " ".join(texts).strip() or None,
                             "transcript_exact": exact, "quality": quality}
        _write_json(out / "profiles.json", profiles)

    # ------------------------------------------------------------------ stage 9
    def _synthesize(self, ws: Path, blocks: list[dict], translations: dict, specs: dict[str, VoiceSpec],
                    locked: set[str], meta: dict, vocals: Path, total_duration: float, lang_name: str,
                    options: JobOptions, out: Path, p: _Progress) -> None:
        """Synthesize every line with the engine of its speaker's voice; one engine loaded at a time.

        Lines are grouped by engine so IndicF5 and Parler never share RAM/VRAM. Timing is decided
        per line exactly as in v2 (fit / atempo / LLM shorten of machine lines / capped overflow)."""
        clips_dir = out / "clips"
        clips_dir.mkdir()
        tts_cache = ws / "tts_cache"
        tts_cache.mkdir(exist_ok=True)
        vocal_sr = sf.info(os.fspath(vocals)).samplerate
        sr = self.s.mix_sample_rate
        thr, max_tempo = self.s.timing_atempo_threshold, self.s.timing_max_atempo
        max_tts_speed = self.s.timing_max_tts_speed
        ordered = sorted(blocks, key=lambda b: (b["start"], b["id"]))
        by_engine: dict[TTSEngine, list[dict]] = {}
        for b in ordered:
            by_engine.setdefault(specs[b["speaker"]].engine, []).append(b)
        segments: list[dict] = []
        report: list[dict] = []
        runs: list[dict] = []
        translator: Optional[Translator] = None
        done = 0

        try:
            for engine_name in sorted(by_engine, key=lambda e: e.value):
                p("synthesize", done / len(ordered), f"Loading {engine_name.value}")
                engine = self._load_engine(engine_name)
                try:
                    md = engine.get_model_metadata() if hasattr(engine, "get_model_metadata") else None
                    model_id = getattr(md, "model_id", None) or engine_name.value
                    model_revision = getattr(md, "model_revision", None)
                    for b in by_engine[engine_name]:
                        idx = ordered.index(b)
                        spk, original = b["speaker"], b["end"] - b["start"]
                        nxt = next((o["start"] for o in ordered[idx + 1:] if o["start"] > b["start"]), total_duration)
                        window_end = b["end"]
                        if nxt > b["end"]:  # natural speech may run into silence, never into the next line
                            window_end = min(b["end"] + self.s.timing_max_overflow_seconds, nxt, total_duration)
                        window = max(original, window_end - b["start"])
                        spec = specs[spk]
                        text = translations[b["id"]]
                        text_locked = b["id"] in locked
                        supports_speed = getattr(getattr(engine, "capabilities", None), "supports_speed", True)
                        history, refines, tts_speed = [], 0, 1.0
                        while True:
                            raw, cache_hit = self._tts_cached(engine, tts_cache, text, spec, b["id"], tts_speed)
                            generated = sf.info(os.fspath(raw)).duration
                            ratio = generated / max(window, 1e-3)
                            history.append({"text": text, "generated": round(generated, 3), "ratio": round(ratio, 4),
                                            **({"tts_speed": tts_speed} if tts_speed != 1.0 else {})})
                            if ratio <= 1.0:
                                tempo, status = 1.0, ("fits" if tts_speed == 1.0 else "tts_speed")
                                break
                            if ratio <= 1.0 + thr:
                                tempo, status = ratio, "atempo"
                                break
                            if refines < self.s.timing_max_refine_attempts and not text_locked:
                                refines += 1
                                target = max(4, int(len(text) / ratio * 0.95))
                                try:
                                    if translator is None:
                                        translator = Translator(self.s, options.translation_engine)
                                    text = translator.shorten(b["text"], text, target, window, lang_name)
                                    log.info("[%s] %.0f%% too long; refined to %d chars (attempt %d)",
                                             b["id"], (ratio - 1) * 100, len(text), refines)
                                    continue
                                except TranslationError as exc:
                                    log.warning("[%s] refine failed: %s", b["id"], exc)
                                    history.append({"refine_error": str(exc)[:300]})
                                    refines = self.s.timing_max_refine_attempts
                            if tts_speed == 1.0 and max_tts_speed > 1.0 and supports_speed:
                                tts_speed = round(min(ratio, max_tts_speed), 3)
                                continue
                            tempo = min(ratio, max_tempo)
                            status = "atempo_capped_overflow" if ratio > max_tempo else "atempo_after_refine"
                            break

                        clip_path = clips_dir / f"{b['id']}.wav"
                        clip = self._render_clip(raw, clip_path, tempo, sr)
                        orig_audio = read_audio(vocals, frames=int(original * vocal_sr), start=int(b["start"] * vocal_sr))[0]
                        gain_db = 0.0
                        orig_db = rms_db(orig_audio.mean(axis=1))
                        if orig_db > -55.0:
                            lim = self.s.loudness_match_max_db
                            gain_db = float(np.clip(orig_db - rms_db(clip), -lim, lim))
                            clip = clip * (10 ** (gain_db / 20))
                        peak = float(np.max(np.abs(clip))) if clip.size else 0.0
                        if peak > 0.99:
                            clip = clip * (0.99 / peak)
                        write_audio(clip_path, clip.astype(np.float32), sr)
                        final_duration = clip.size / sr
                        placement_end = b["start"] + final_duration
                        overlaps_next = placement_end > nxt + 0.05 and nxt < total_duration
                        warning = None
                        if status == "atempo_capped_overflow":
                            warning = (f"Line is {final_duration - window:.2f}s longer than its slot after "
                                       f"{max_tempo:.2f}x speed-up; review the translation.")
                        elif overlaps_next:
                            warning = "Dubbed line runs into the next line."

                        m = meta.get(spk, {})
                        segments.append(DialogueSegment(
                            segment_id=b["id"], start_time=b["start"], end_time=b["end"], speaker_id=spk,
                            original_text=b["text"], translated_text=text,
                            generated_audio_path=clip_path.relative_to(out).as_posix(),
                            original_duration=round(original, 3), generated_duration=round(final_duration, 3),
                            speed_ratio=round(tempo, 4),
                            speaker_display_name=m.get("name"), voice_source=m.get("source"),
                        ).model_dump())
                        report.append({"segment_id": b["id"], "speaker_id": spk, "status": status,
                                       "refine_attempts": refines, "text_locked": text_locked,
                                       "original_duration": round(original, 3), "window_seconds": round(window, 3),
                                       "generated_duration": round(generated, 3), "final_duration": round(final_duration, 3),
                                       "speed_adjustment": round(tempo, 4), "tts_speed": tts_speed,
                                       "placement_start": round(b["start"], 3), "placement_end": round(placement_end, 3),
                                       "overflow_seconds": round(max(0.0, final_duration - original), 3),
                                       "overlaps_next": overlaps_next, "warning": warning,
                                       "gain_db": round(gain_db, 2), "attempts": history})
                        runs.append({"segment_id": b["id"], "speaker_id": spk, "character_id": m.get("character_id"),
                                     "voice_profile_id": m.get("voice_profile_id"), "voice_id": m.get("voice_id"),
                                     "engine": engine_name.value, "model_id": model_id, "model_revision": model_revision,
                                     "generation_parameters": spec.generation_parameters or {},
                                     "text_hash": text_hash(text),
                                     "reference_hash": self._digest(spec.reference_path) if spec.reference_path else None,
                                     "voice_prompt": spec.voice_prompt, "seed": spec.seed,
                                     "random_seed": int(hashlib.sha256(b["id"].encode()).hexdigest()[:8], 16),
                                     "generated_audio_path": clip_path.relative_to(out).as_posix(),
                                     "generation_status": "cached" if cache_hit else "generated",
                                     "timing_status": status, "qa_status": "PASS" if not warning else "WARNING"})
                        done += 1
                        p("synthesize", done / len(ordered), f"Hindi line {done}/{len(ordered)} ({engine_name.value})")
                finally:
                    engine.close()
                    _free_memory()
        finally:
            if translator:
                translator.close()

        segments.sort(key=lambda s_: (s_["start_time"], s_["segment_id"]))
        report.sort(key=lambda r: r["placement_start"])
        overflow = [r for r in report if r["status"] == "atempo_capped_overflow"]
        if overflow:
            log.warning("%d segment(s) still exceed their slot after refinement and %.2fx tempo; see timing_report.json",
                        len(overflow), max_tempo)
        _write_json(out / "segments.json", segments)
        _write_json(out / "generation_runs.json", runs)
        _write_json(out / "timing_report.json", {
            "summary": {status: sum(1 for r in report if r["status"] == status)
                        for status in ("fits", "tts_speed", "atempo", "atempo_after_refine", "atempo_capped_overflow")},
            "segments": report,
        })

    def _tts_cached(self, engine: BaseTTSEngine, cache: Path, text: str, spec: VoiceSpec, seg_id: str,
                    speed: float = 1.0) -> tuple[Path, bool]:
        """Per-line cache keyed by *everything* that changes the audio. Returns (path, was_cache_hit).

        Cache entries include the complete engine, model, voice, and generation configuration."""
        ref_digest = self._digest(spec.reference_path) if spec.reference_path else None
        key_src: dict = {"text": text, "voice": spec.cache_fingerprint(ref_digest), "engine": engine.fingerprint, "v": 2}
        if speed != 1.0:
            key_src["speed"] = speed
        key = hashlib.sha256(json.dumps(key_src, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
        path = cache / f"{key}.wav"
        if path.is_file():
            return path, True
        seed = int(hashlib.sha256(seg_id.encode()).hexdigest()[:8], 16)
        extra = {"speed": speed} if speed != 1.0 else {}
        for attempt in range(3):
            try:
                wav = engine.synthesize(text, spec, seed + attempt, **extra)
            except TTSEngineError as exc:
                raise PipelineError(f"{spec.engine.value} failed on segment {seg_id}: {exc}") from exc
            wav = trim_silence(np.asarray(wav, dtype=np.float32).reshape(-1), engine.sample_rate)
            problems = engine.validate_output(wav, text) if hasattr(engine, "validate_output") else []
            if wav.size >= int(0.1 * engine.sample_rate) and not problems:
                return write_audio(path, wav, engine.sample_rate), False
            log.warning("[%s] %s returned unusable audio (attempt %d: %s); retrying with a new seed", seg_id,
                        spec.engine.value, attempt + 1, "; ".join(problems) or "silence")
        raise PipelineError(f"{spec.engine.value} produced no usable audio for segment {seg_id}: {text!r}")

    def _render_clip(self, raw: Path, dst: Path, tempo: float, sr: int) -> np.ndarray:
        src = raw
        if tempo > 1.0001:
            src = self.ff.atempo(raw, dst.with_name(dst.stem + ".tempo.wav"), tempo)
        self.ff.resample(src, dst, sr, 1, codec="pcm_f32le")
        if src != raw:
            src.unlink(missing_ok=True)
        return read_audio(dst)[0][:, 0]

    # ------------------------------------------------------------------ stage 10
    def _mix(self, segments: list[DialogueSegment], mode: MixMode, master: Path, background: Optional[Path],
             out: Path, p: _Progress) -> None:
        """Dialogue-aware mix. Writes ``dialogue.wav`` (placed Hindi lines), ``bed.wav`` (the processed bed:
        separated background, ducked original or silence), ``mix.wav`` and ``mix_report.json``."""
        info = sf.info(os.fspath(master))
        sr, channels, total = info.samplerate, info.channels, info.frames
        if sr != self.s.mix_sample_rate:
            raise PipelineError(f"Master sample rate {sr} != MIX_SAMPLE_RATE {self.s.mix_sample_rate}")
        bed_src = {MixMode.PRESERVE_BACKGROUND: background, MixMode.ORIGINAL_PLUS_DUB: master,
                   MixMode.DUB_ONLY: None}[mode]
        clips = []
        for seg in segments:
            s_ = int(round(seg.start_time * sr))
            n = sf.info(seg.generated_audio_path).frames
            clips.append((s_, s_ + n, seg.generated_audio_path))
        clips.sort()
        block = 60 * sr

        # pass 1: the dialogue stem ----------------------------------------------------
        p("mix", 0.0, "Placing Hindi dialogue")
        loaded: dict[str, np.ndarray] = {}
        with sf.SoundFile(os.fspath(out / "dialogue.wav"), "w", sr, 1, "FLOAT") as fd:
            for b0 in range(0, total, block):
                b1 = min(total, b0 + block)
                dialogue = np.zeros(b1 - b0, dtype=np.float32)
                for s_, e, path in clips:
                    if s_ >= b1 or e <= b0:
                        continue
                    if path not in loaded:
                        loaded[path] = read_audio(path)[0][:, 0]
                    lo, hi = max(s_, b0), min(e, b1)
                    dialogue[lo - b0:hi - b0] += loaded[path][lo - s_:hi - s_]
                for path in {pth for s_, e, pth in clips if e <= b1 and pth in loaded}:
                    del loaded[path]
                fd.write(dialogue)

        # loudness plan (static gains) -----------------------------------------------
        def _lufs(path: Optional[Path]) -> Optional[float]:
            if path is None:
                return None
            try:
                return self.ff.measure_loudness(path).get("integrated_lufs")
            except MediaError as exc:
                log.warning("Loudness measurement failed for %s: %s", path.name, exc)
                return None

        gains = plan_gains(_lufs(out / "dialogue.wav"), _lufs(bed_src), self.s.dialogue_target_lufs,
                           self.s.dialogue_normalize_max_db, self.s.min_dialogue_bed_margin_db, bed_src is not None)
        for note in gains.notes:
            log.info("mix: %s", note)
        duck_db = self.s.duck_db if mode == MixMode.ORIGINAL_PLUS_DUB else self.s.music_duck_db
        dynamic = self.s.dynamic_ducking_enabled or mode == MixMode.ORIGINAL_PLUS_DUB
        follower = DuckingFollower(sr, duck_db, self.s.duck_attack_ms, self.s.duck_release_ms) if dynamic else None
        legacy_gain, legacy_ramp = 10 ** (self.s.duck_db / 20), int(0.15 * sr)

        # pass 2: bed + mix -----------------------------------------------------------
        mix_path = out / "mix.wav"
        peak_before_limiter = 0.0
        with sf.SoundFile(os.fspath(mix_path), "w", sr, channels, "FLOAT") as fw, \
                sf.SoundFile(os.fspath(out / "bed.wav"), "w", sr, channels, "FLOAT") as fb, \
                sf.SoundFile(os.fspath(out / "dialogue.wav"), "r") as fd:
            for b0 in range(0, total, block):
                b1 = min(total, b0 + block)
                n = b1 - b0
                dialogue = fd.read(n, dtype="float32", always_2d=True)[:, 0]
                if dialogue.shape[0] < n:
                    dialogue = np.pad(dialogue, (0, n - dialogue.shape[0]))
                dialogue = apply_gain_db(dialogue, gains.dialogue_db)
                if bed_src is not None:
                    bed = read_audio(bed_src, frames=n, start=b0)[0]
                    if bed.shape[0] < n:
                        bed = np.pad(bed, ((0, n - bed.shape[0]), (0, 0)))
                    if bed.shape[1] != channels:
                        bed = np.repeat(bed.mean(axis=1, keepdims=True), channels, axis=1)
                    bed = apply_gain_db(bed, gains.bed_db)
                    if follower is not None:
                        bed = bed * follower.process(dialogue)[:, None]
                    elif mode == MixMode.ORIGINAL_PLUS_DUB:
                        bed = bed * ducking_envelope(b0, b1, clips, legacy_gain, legacy_ramp)[:, None]
                else:
                    bed = np.zeros((n, channels), dtype=np.float32)
                fb.write(bed.astype(np.float32))
                mixed = (bed + dialogue[:, None]).astype(np.float32)
                peak_before_limiter = max(peak_before_limiter,
                                          float(np.max(np.abs(mixed))) if mixed.size else 0.0)
                fw.write(mixed)
                p("mix", b1 / total, "Mixing")
        peak_ceiling_dbfs = self.s.true_peak_ceiling_dbtp - 0.5
        peak_ceiling = 10 ** (peak_ceiling_dbfs / 20)
        peak_gain = min(1.0, peak_ceiling / peak_before_limiter) if peak_before_limiter else 1.0
        if peak_gain < 1.0:
            limited_path = out / "mix.limited.wav"
            with sf.SoundFile(os.fspath(mix_path), "r") as source, \
                    sf.SoundFile(os.fspath(limited_path), "w", sr, channels, "FLOAT") as target:
                while True:
                    mixed = source.read(block, dtype="float32", always_2d=True)
                    if not mixed.size:
                        break
                    target.write((mixed * peak_gain).astype(np.float32))
            limited_path.replace(mix_path)
        _write_json(out / "mix_report.json", {
            "mode": mode.value, "bed": "background" if mode == MixMode.PRESERVE_BACKGROUND else
            ("original" if mode == MixMode.ORIGINAL_PLUS_DUB else None),
            "dialogue_gain_db": gains.dialogue_db, "bed_gain_db": gains.bed_db,
            "dialogue_lufs_before": gains.dialogue_lufs_before, "bed_lufs_before": gains.bed_lufs_before,
            "dynamic_ducking": follower is not None, "duck_db": duck_db if bed_src is not None else None,
            "attack_ms": self.s.duck_attack_ms, "release_ms": self.s.duck_release_ms,
            "peak_protection": {"input_peak": round(peak_before_limiter, 4),
                                "output_peak": round(peak_before_limiter * peak_gain, 4),
                                "ceiling_dbfs": round(peak_ceiling_dbfs, 2),
                                "gain_db": round(20 * np.log10(peak_gain), 2) if peak_gain < 1.0 else 0.0},
            "target_lufs": self.s.dialogue_target_lufs, "notes": gains.notes})

    # ------------------------------------------------------------------ stage 11
    def _validate_output(self, output: Path, source: MediaInfo) -> None:
        got = self.ff.probe(output)
        tolerance = max(0.5, 0.01 * source.duration)
        if abs(got.duration - source.duration) > tolerance:
            raise PipelineError(f"Output duration {got.duration:.2f}s differs from source {source.duration:.2f}s.")
        if got.audio_streams != 1:
            raise PipelineError(f"Output has {got.audio_streams} audio streams; expected exactly 1.")
        log.info("Output validated: %s (%.1fs)", output.name, got.duration)


# =============================================================================== pure helpers
def build_dialogue_blocks(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Intersect Whisper words with diarization turns into speaker-attributed dialogue blocks.

    Each word gets the speaker whose turn overlaps it most (nearest turn if none overlaps). Within a
    Whisper segment, consecutive words of one speaker form a block; blips shorter than 0.4 s / two
    words are absorbed by a neighbour because word-level diarization jitter is common.
    """
    if turns:
        starts = np.array([t["start"] for t in turns])
        ends = np.array([t["end"] for t in turns])
        labels = [t["speaker"] for t in turns]

    def speaker_for(ws: float, we: float) -> str:
        if not turns:
            return "SPEAKER_00"
        overlap = np.minimum(ends, we) - np.maximum(starts, ws)
        i = int(np.argmax(overlap))
        if overlap[i] > 0:
            return labels[i]
        gap = np.maximum(starts - we, ws - ends)
        return labels[int(np.argmin(gap))]

    def short(g: dict) -> bool:
        return len(g["words"]) <= 2 and g["words"][-1]["end"] - g["words"][0]["start"] < 0.4

    blocks: list[dict] = []
    for seg in segments:
        words = [w for w in seg.get("words") or [] if w["word"].strip()]
        if not words:
            words = [{"start": seg["start"], "end": seg["end"], "word": seg["text"]}]
        groups: list[dict] = []
        for w in words:
            spk = speaker_for(w["start"], w["end"])
            if groups and groups[-1]["speaker"] == spk:
                groups[-1]["words"].append(w)
            else:
                groups.append({"speaker": spk, "words": [w]})
        if len(groups) > 1:
            smoothed: list[dict] = []
            for g in groups:
                if smoothed and short(g):
                    smoothed[-1]["words"] += g["words"]
                else:
                    smoothed.append(g)
            if len(smoothed) > 1 and short(smoothed[0]):
                smoothed[1]["words"] = smoothed[0]["words"] + smoothed[1]["words"]
                smoothed.pop(0)
            groups = []
            for g in smoothed:
                if groups and groups[-1]["speaker"] == g["speaker"]:
                    groups[-1]["words"] += g["words"]
                else:
                    groups.append(g)
        for g in groups:
            text = "".join(w["word"] for w in g["words"]).strip()
            if not text:
                continue
            start, end = g["words"][0]["start"], g["words"][-1]["end"]
            blocks.append({"start": round(start, 3), "end": round(max(end, start + 0.2), 3),
                           "speaker": g["speaker"], "text": text})
    blocks.sort(key=lambda b: b["start"])
    for i, b in enumerate(blocks, 1):
        b["id"] = f"seg_{i:05d}"
    return blocks


def select_reference_spans(blocks: list[dict], turns: list[dict], speaker: str,
                           target_seconds: float) -> list[list[float]]:
    """Pick up to ``target_seconds`` of the speaker's longest solo (non-overlapped) speech."""
    others = [(t["start"], t["end"]) for t in turns if t["speaker"] != speaker]

    def overlap_with_others(s: float, e: float) -> float:
        return sum(max(0.0, min(e, oe) - max(s, os_)) for os_, oe in others if os_ < e and oe > s)

    own = [(b["start"], b["end"]) for b in blocks if b["speaker"] == speaker]
    clean = [(s, e) for s, e in own if e - s >= 1.0 and overlap_with_others(s, e) < 0.05]
    pool = sorted(clean or own, key=lambda se: se[1] - se[0], reverse=True)
    chosen, total = [], 0.0
    for s, e in pool:
        take = min(e - s, target_seconds - total)
        if take < 0.5:
            break
        chosen.append([round(s, 3), round(s + take, 3)])
        total += take
        if total >= target_seconds - 1e-3:
            break
    return sorted(chosen)


def usable_reference_seconds(blocks: list[dict], turns: list[dict], speaker: str) -> float:
    """Total clean solo speech (>= 1 s lines without overlap from other speakers) a speaker has in the video."""
    others = [(t["start"], t["end"]) for t in turns if t["speaker"] != speaker]
    total = 0.0
    for b in blocks:
        if b["speaker"] != speaker or b["end"] - b["start"] < 1.0:
            continue
        overlap = sum(max(0.0, min(b["end"], oe) - max(b["start"], os_)) for os_, oe in others
                      if os_ < b["end"] and oe > b["start"])
        if overlap < 0.05:
            total += b["end"] - b["start"]
    return total


def ducking_envelope(b0: int, b1: int, clips: list[tuple[int, int, str]], gain: float, ramp: int) -> np.ndarray:
    """Gain curve for samples [b0, b1): ``gain`` under dialogue, linear ramps of ``ramp`` samples around it."""
    env = np.ones(b1 - b0, dtype=np.float32)
    t = np.arange(b0, b1, dtype=np.int64)
    for s, e, _ in clips:
        if s - ramp >= b1 or e + ramp <= b0:
            continue
        distance = np.maximum(np.maximum(s - t, t - (e - 1)), 0)
        frac = np.clip(distance / max(ramp, 1), 0.0, 1.0)
        env = np.minimum(env, gain + (1.0 - gain) * frac)
    return env


@contextlib.contextmanager
def _trusted_checkpoint_loading() -> Iterator[None]:
    """pyannote 3.x checkpoints pickle non-tensor objects; torch>=2.6 defaults torch.load to weights_only.

    Scoped to loading the official pyannote checkpoints only; call sites that pass weights_only
    explicitly are unaffected.
    """
    previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous


_DLLS_REGISTERED = False


def _register_torch_cuda_dlls() -> None:
    """ctranslate2 (faster-whisper) needs cuBLAS/cuDNN DLLs; reuse the ones shipped inside the torch wheel."""
    global _DLLS_REGISTERED
    if sys.platform != "win32" or _DLLS_REGISTERED:
        return
    import torch

    lib = Path(torch.__file__).parent / "lib"
    if lib.is_dir():
        os.add_dll_directory(os.fspath(lib))
        os.environ["PATH"] = os.fspath(lib) + os.pathsep + os.environ.get("PATH", "")
    _DLLS_REGISTERED = True


def _free_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "video"


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data) -> None:
    tmp = path.with_name(f"{path.name}.part-{uuid.uuid4().hex[:8]}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
