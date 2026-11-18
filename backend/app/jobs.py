# hindi-dubbing-platform/backend/app/jobs.py

"""In-process job manager: a bounded ThreadPoolExecutor plus JSON job records on disk.

Records are persisted atomically on every state change, so status survives a server restart.
Jobs that were queued or running when the server stopped are re-queued at startup; because every
pipeline stage is cached, a re-queued job resumes from the last completed stage.

Two job kinds share this manager:

* single-shot jobs (``POST /api/v1/jobs``): queued -> running -> succeeded | failed   (unchanged)
* project jobs with review: queued -> running (analysis) -> awaiting_review
  -> [confirm] -> generating -> succeeded | failed.  ``cancelled`` is reachable from every
  non-terminal state; failed/cancelled jobs can be retried from their last phase.

Heavy work only ever runs on the worker threads; HTTP handlers only read/write records.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

import numpy as np

from backend.app.config import MixMode, Settings, TranslationEngine
from backend.services import drafts as dr
from backend.services import speaker_identity as si
from backend.services.media import trim_silence, write_audio
from backend.services.pipeline import DubbingPipeline, GenerationPlan, JobOptions, PipelineError
from backend.services.projects import ProjectStore
from backend.services.series import EpisodeSpeakerMapping, MappingStatus, SeriesStore, SpeakerMapping
from backend.services.storage import ConflictError, NotFoundError, ValidationFailed, atomic_write_json, read_json
from backend.services.tts import TTSEngineError
from backend.services.voice_library import VoiceLibrary
from backend.services.voices import CharacterVoice

log = logging.getLogger("dubbing.jobs")
current_job_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_job_id", default=None)


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    GENERATING = "generating"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}
ACTIVE = {JobState.QUEUED, JobState.RUNNING, JobState.GENERATING}
_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.QUEUED: {JobState.RUNNING, JobState.FAILED, JobState.CANCELLED},
    JobState.RUNNING: {JobState.AWAITING_REVIEW, JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED,
                       JobState.QUEUED},
    JobState.AWAITING_REVIEW: {JobState.GENERATING, JobState.CANCELLED},
    JobState.GENERATING: {JobState.GENERATING, JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED},
    JobState.FAILED: {JobState.QUEUED, JobState.GENERATING},
    JobState.CANCELLED: {JobState.QUEUED, JobState.GENERATING},
    JobState.SUCCEEDED: set(),
}


class InvalidTransition(ConflictError):
    pass


class JobCancelled(Exception):
    pass


class JobOptionsModel(BaseModel):
    source_language: Optional[str] = None
    mix_mode: MixMode = MixMode.PRESERVE_BACKGROUND
    translation_engine: TranslationEngine = TranslationEngine.OPENROUTER
    min_speakers: Optional[int] = Field(None, ge=1, le=20)
    max_speakers: Optional[int] = Field(None, ge=1, le=20)


class StageEvent(BaseModel):
    stage: str
    phase: str
    status: str = "running"  # running | completed | failed | cancelled
    started_at: datetime
    finished_at: Optional[datetime] = None
    message: str = ""


class JobRecord(BaseModel):
    job_id: str
    state: JobState = JobState.QUEUED
    stage: str = "queued"
    progress: float = 0.0
    message: str = ""
    input_filename: str
    input_path: str
    options: JobOptionsModel
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    output_path: Optional[str] = None
    manifest_path: Optional[str] = None
    timing_report_path: Optional[str] = None
    source_language: Optional[str] = None
    speakers: Optional[int] = None
    segments: Optional[int] = None
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    # project / review workflow (all optional: single-shot records keep their old shape)
    project_id: Optional[str] = None
    video_id: Optional[str] = None
    review_required: bool = False
    phase: str = "analysis"  # analysis | generation
    approved_version: Optional[int] = None
    stage_history: list[StageEvent] = Field(default_factory=list)
    analysis_completed_at: Optional[datetime] = None
    generation_started_at: Optional[datetime] = None
    # v3: quality gate (optional: v2 records load unchanged)
    qa_status: Optional[str] = None            # PASS | WARNING | REVIEW_REQUIRED | FAILED
    qa_report_path: Optional[str] = None
    release_approved: bool = False
    release_note: str = ""
    engines: list[str] = Field(default_factory=list)


class _JobLogFilter(logging.Filter):
    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id

    def filter(self, record: logging.LogRecord) -> bool:
        return current_job_id.get() == self.job_id


class JobManager:
    def __init__(self, settings: Settings, projects: Optional[ProjectStore] = None,
                 voice_library: Optional[VoiceLibrary] = None, series: Optional[SeriesStore] = None):
        self.s = settings
        self.pipeline = DubbingPipeline(settings)
        self.projects = projects
        self.voice_library = voice_library
        self.series = series
        self._auditions: dict[str, dict] = {}
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._active: set[str] = set()
        self._cancel: dict[str, threading.Event] = {}
        self._executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_jobs, thread_name_prefix="dub-worker")
        self._load_and_resume()

    # ------------------------------------------------------------------ public
    def submit(self, input_path: Path, input_filename: str, options: JobOptionsModel, job_id: str,
               project_id: Optional[str] = None, video_id: Optional[str] = None,
               review_required: bool = False) -> JobRecord:
        record = JobRecord(job_id=job_id, input_filename=input_filename, input_path=os.fspath(input_path),
                           options=options, created_at=_now(), message="Waiting for a worker",
                           project_id=project_id, video_id=video_id, review_required=review_required)
        with self._lock:
            self._jobs[job_id] = record
            self._persist(record)
        self._schedule(job_id)
        log.info("Job %s queued (%s)", job_id, input_filename)
        return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            rec = self._jobs.get(job_id)
            return rec.model_copy(deep=True) if rec else None

    def list(self, project_id: Optional[str] = None) -> list[JobRecord]:
        with self._lock:
            recs = [r.model_copy(deep=True) for r in self._jobs.values()
                    if project_id is None or r.project_id == project_id]
        return sorted(recs, key=lambda r: r.created_at, reverse=True)

    def counts(self, project_id: Optional[str] = None) -> dict[str, int]:
        with self._lock:
            out = {s.value: 0 for s in JobState}
            for r in self._jobs.values():
                if project_id is None or r.project_id == project_id:
                    out[r.state.value] += 1
            return out

    def shutdown(self) -> None:
        for ev in self._cancel.values():
            ev.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def new_job_id() -> str:
        return uuid.uuid4().hex

    def job_output_dir(self, job_id: str) -> Path:
        return self.s.outputs_dir / job_id

    def analysis_index(self, job_id: str) -> dict:
        f = self.job_output_dir(job_id) / "analysis.json"
        if not f.is_file():
            raise NotFoundError("Analysis artifacts are not available for this job")
        return read_json(f)

    # ------------------------------------------------------------------ review workflow
    def _review_record(self, job_id: str) -> JobRecord:
        rec = self.get(job_id)
        if rec is None:
            raise NotFoundError("Job not found")
        if not rec.project_id or not rec.video_id or self.projects is None:
            raise NotFoundError("This job has no review draft (single-shot job)")
        return rec

    def load_draft(self, job_id: str) -> dr.DraftManifest:
        rec = self._review_record(job_id)
        draft = self.projects.load_draft(rec.project_id, rec.video_id)
        if draft.job_id != job_id:
            raise NotFoundError("This job's draft was superseded by a newer job for the same video")
        return draft

    def update_draft(self, job_id: str, patch: dr.DraftPatch) -> dr.DraftManifest:
        with self._lock:
            rec = self._review_record(job_id)
            if rec.state != JobState.AWAITING_REVIEW:
                raise ConflictError(f"Job is {rec.state.value}; the draft can only be edited while awaiting review.")
            draft = self.load_draft(job_id)
            profiles = self._profiles(job_id)
            self._check_patch_references(rec.project_id, patch)
            updated = dr.apply_patch(draft, patch, profiles, self.s)
            self._decorate_identity(rec.project_id, updated)
            self.projects.save_draft(updated)
            return updated

    def _check_patch_references(self, project_id: str, patch: dr.DraftPatch) -> None:
        errors = []
        for sp in patch.speakers:
            if sp.voice_id:
                if self.voice_library is None or self.voice_library.lookup(sp.voice_id) is None:
                    errors.append(f"{sp.speaker_id}: voice '{sp.voice_id}' is not an approved, active Voice Library voice")
            if sp.character_id:
                try:
                    if self.series is None:
                        raise NotFoundError("series disabled")
                    self.series.get_character(project_id, sp.character_id)
                except NotFoundError:
                    errors.append(f"{sp.speaker_id}: character does not belong to this project")
            if sp.engine is not None and sp.engine not in self.s.enabled_tts_engines:
                errors.append(f"{sp.speaker_id}: engine '{sp.engine.value}' is not enabled")
        if errors:
            raise ValidationFailed("Draft update rejected", errors)

    def _decorate_identity(self, project_id: str, draft: dr.DraftManifest) -> None:
        """Fill read-only character names on the draft speakers."""
        if self.series is None:
            return
        names = {c.character_id: c.canonical_name for c in self.series.list_characters(project_id, include_archived=True)}
        for spk in draft.speakers:
            spk.character_name = names.get(spk.character_id or "")
            spk.suggested_character_name = names.get(spk.suggested_character_id or "")

    # ------------------------------------------------------------------ series / identity
    def cast_ids(self) -> set[str]:
        if self.voice_library is None:
            return set()
        return {v.voice_id for v in self.voice_library.list() if v.selectable}

    def character_lookup(self, project_id: str):
        """character_id -> CharacterVoice for characters with an approved active voice profile."""
        def lookup(character_id: str) -> Optional[CharacterVoice]:
            if self.series is None:
                return None
            try:
                c = self.series.get_character(project_id, character_id)
            except NotFoundError:
                return None
            prof = c.active_profile()
            if prof is None or not prof.approved:
                return None
            return CharacterVoice(character_id=c.character_id, character_name=c.canonical_name,
                                  voice_profile_id=prof.voice_profile_id, voice_source=prof.voice_source,
                                  engine=prof.engine, voice_asset_id=prof.voice_asset_id)
        return lookup

    def characters_with_voice(self, project_id: str) -> set[str]:
        if self.series is None:
            return set()
        look = self.character_lookup(project_id)
        return {c.character_id for c in self.series.list_characters(project_id) if look(c.character_id)}

    def _episode_embeddings(self, job_id: str) -> dict[str, np.ndarray]:
        index = self.analysis_index(job_id)
        f = index.get("embeddings")
        if not f or not Path(f).is_file():
            return {}
        return {spk: np.asarray(v, dtype=np.float32) for spk, v in read_json(f).get("speakers", {}).items()}

    def suggest_identities(self, job_id: str, draft: dr.DraftManifest, project_id: str) -> dr.DraftManifest:
        """Attach cross-episode character suggestions (never assignments) to the draft speakers."""
        if self.series is None:
            return draft
        profiles = self._profiles(job_id)
        seconds = {lbl: float((prof or {}).get("available_seconds", 0.0)) for lbl, prof in profiles.items()}
        embeddings = self._episode_embeddings(job_id)
        bank = self.series.character_embeddings(project_id)
        suggestions = [s.public() for s in si.match_speakers(embeddings, seconds, bank, self.s)]
        if not embeddings:
            for s_ in suggestions:
                s_["reason"] = s_["reason"] or "speaker embeddings unavailable"
        names = {c.character_id: c.canonical_name for c in self.series.list_characters(project_id, include_archived=True)}
        history = {cid: self.series.character_history(project_id, cid)
                   for cid in {s_["character_id"] for s_ in suggestions if s_.get("character_id")}}
        atomic_write_json(self.job_output_dir(job_id) / "identity_suggestions.json",
                          {"job_id": job_id, "suggestions": suggestions, "characters_with_evidence": sorted(bank)})
        draft = dr.attach_identity(draft, suggestions, names, history)
        self._decorate_identity(project_id, draft)
        return dr.refresh(draft, profiles, self.s)

    def _record_episode_mapping(self, rec: JobRecord, approved: dr.DraftManifest) -> None:
        if self.series is None:
            return
        try:
            video = self.projects.get_video(rec.project_id, rec.video_id)
            episode_number = video.episode_number
        except NotFoundError:
            episode_number = None
        embeddings = self._episode_embeddings(rec.job_id)
        mappings = []
        for spk in approved.speakers:
            if spk.segment_count == 0:
                continue
            status = MappingStatus.CONFIRMED if spk.character_id else (
                MappingStatus(spk.mapping_status) if spk.mapping_status in ("rejected", "suggested") else MappingStatus.UNASSIGNED)
            mappings.append(SpeakerMapping(
                local_speaker_id=spk.speaker_id, character_id=spk.character_id,
                confidence=spk.identity_confidence, mapping_status=status, reviewed_by_user=bool(spk.character_id),
                evidence={"origin_labels": spk.origin_labels, "speech_seconds": spk.usable_reference_duration,
                          "reason": spk.identity_reason, "candidates": spk.identity_candidates[:3]}))
            if spk.character_id:
                vecs = [embeddings[lbl] for lbl in spk.origin_labels if lbl in embeddings]
                if vecs:
                    self.series.add_character_embedding(
                        rec.project_id, spk.character_id, np.mean(vecs, axis=0),
                        {"job_id": rec.job_id, "video_id": rec.video_id, "local_speaker_id": spk.speaker_id,
                         "episode_number": episode_number, "speech_seconds": spk.usable_reference_duration})
        self.series.write_episode_mapping(EpisodeSpeakerMapping(
            project_id=rec.project_id, video_id=rec.video_id, job_id=rec.job_id, episode_number=episode_number,
            approved_version=approved.version, mappings=mappings, created_at=_now()))

    def confirm(self, job_id: str, base_version: int) -> tuple[JobRecord, list[str]]:
        """Validate, lock an immutable approved snapshot and start generation exactly once."""
        with self._lock:
            rec = self._review_record(job_id)
            if rec.state in (JobState.GENERATING, JobState.SUCCEEDED) and rec.approved_version:
                raise ConflictError(f"Already confirmed (approved version {rec.approved_version}); job is {rec.state.value}.")
            self._check_transition(rec.state, JobState.GENERATING)
            draft = self.load_draft(job_id)
            if draft.version != base_version:
                raise ConflictError(f"Draft was modified (current version {draft.version}, yours {base_version}); reload.")
            errors, warnings = dr.validate_for_confirm(
                draft, self.projects.library_ids_with_transcript(rec.project_id), self.pipeline.bank, self.s,
                cast_ids=self.cast_ids(), character_ids=self.characters_with_voice(rec.project_id),
                profiles=self._profiles(job_id))
            if errors:
                raise ValidationFailed("Draft is not ready for generation", errors)
            approved = draft.model_copy(update={"review_status": "approved", "approved_version": draft.version,
                                                "approved_at": _now()})
            self.projects.write_approved_snapshot(approved)
            self.projects.save_draft(approved)
            self._record_episode_mapping(rec, approved)
            self._update(job_id, state=JobState.GENERATING, phase="generation", approved_version=draft.version,
                         stage="queued", progress=0.0, message="Approved; waiting for a worker", error=None, finished_at=None,
                         generation_started_at=_now(), options=rec.options.model_copy(update={"mix_mode": draft.mix_mode}))
        self._schedule(job_id)
        log.info("Job %s confirmed at draft version %d", job_id, draft.version)
        return self.get(job_id), warnings

    def cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            rec = self.get(job_id)
            if rec is None:
                raise NotFoundError("Job not found")
            self._check_transition(rec.state, JobState.CANCELLED)
            if job_id in self._active:
                self._cancel.setdefault(job_id, threading.Event()).set()
                self._update(job_id, message="Cancellation requested; stopping after the current step")
            else:
                self._update(job_id, state=JobState.CANCELLED, finished_at=_now(), message="Cancelled")
        return self.get(job_id)

    def retry(self, job_id: str) -> JobRecord:
        with self._lock:
            rec = self.get(job_id)
            if rec is None:
                raise NotFoundError("Job not found")
            if rec.state not in (JobState.FAILED, JobState.CANCELLED):
                raise InvalidTransition(f"Only failed or cancelled jobs can be retried (job is {rec.state.value}).")
            if not Path(rec.input_path).is_file():
                raise ConflictError("The input video no longer exists; upload it again.")
            target = JobState.GENERATING if (rec.phase == "generation" and rec.approved_version) else JobState.QUEUED
            self._update(job_id, state=target, error=None, finished_at=None, stage="queued",
                         message="Retry queued (completed stages are reused from cache)")
        self._schedule(job_id)
        return self.get(job_id)

    def delete_records(self, job_ids: list[str]) -> None:
        import shutil
        with self._lock:
            for jid in job_ids:
                self._jobs.pop(jid, None)
                (self.s.jobs_dir / f"{jid}.json").unlink(missing_ok=True)
                shutil.rmtree(self.job_output_dir(jid), ignore_errors=True)

    def jobs_using_library_speaker(self, project_id: str, speaker_id: str) -> list[str]:
        """Non-terminal jobs whose draft selects this library speaker (blocks deletion)."""
        hits = []
        for rec in self.list(project_id):
            if rec.state in TERMINAL or not rec.video_id:
                continue
            try:
                draft = self.projects.load_draft(project_id, rec.video_id)
            except NotFoundError:
                continue
            if any(s.voice_source == dr.VoiceSource.LIBRARY and s.library_speaker_id == speaker_id
                   for s in draft.speakers):
                hits.append(rec.job_id)
        return hits

    def set_release(self, job_id: str, approved: bool, note: str = "") -> JobRecord:
        with self._lock:
            rec = self.get(job_id)
            if rec is None:
                raise NotFoundError("Job not found")
            if rec.state != JobState.SUCCEEDED:
                raise ConflictError(f"Job is {rec.state.value}; only a finished dub can be released.")
            if approved and rec.qa_status == "FAILED":
                raise ConflictError("QA status is FAILED; fix the reported problems and regenerate before release.")
            self._update(job_id, release_approved=approved, release_note=note.strip())
        return self.get(job_id)

    # ------------------------------------------------------------------ voice auditions
    def start_audition(self, voice_id: str, text: str) -> dict:
        """Generate a preview for a Voice Library voice on the worker pool (same resource budget as jobs)."""
        if self.voice_library is None:
            raise NotFoundError("Voice Library is not enabled")
        voice = self.voice_library.get(voice_id)
        if voice.engine not in self.s.enabled_tts_engines:
            raise ConflictError(f"Engine '{voice.engine.value}' is not enabled (TTS_ENGINES_ENABLED).")
        audition_id = uuid.uuid4().hex
        entry = {"audition_id": audition_id, "voice_id": voice_id, "text": text.strip(), "status": "queued",
                 "error": None, "created_at": _now(), "finished_at": None}
        with self._lock:
            self._auditions[audition_id] = entry
        self._executor.submit(self._run_audition, audition_id)
        return dict(entry)

    def get_audition(self, audition_id: str) -> Optional[dict]:
        with self._lock:
            e = self._auditions.get(audition_id)
            return dict(e) if e else None

    def _run_audition(self, audition_id: str) -> None:
        with self._lock:
            entry = self._auditions[audition_id]
            entry["status"] = "running"
        voice_id, text = entry["voice_id"], entry["text"]
        try:
            voice = self.voice_library.get(voice_id)
            spec = self.voice_library.to_spec(voice)
            engine = self.pipeline._load_engine(voice.engine)
            try:
                md = engine.get_model_metadata() if hasattr(engine, "get_model_metadata") else None
                wav = trim_silence(np.asarray(engine.synthesize(text, spec, 7), dtype=np.float32).reshape(-1),
                                   engine.sample_rate)
                problems = engine.validate_output(wav, text) if hasattr(engine, "validate_output") else []
                tmp = self.voice_library.voice_dir(voice_id) / f"preview.{audition_id[:8]}.wav"
                write_audio(tmp, wav, engine.sample_rate)
                self.voice_library.store_preview(
                    voice_id, tmp, text,
                    {"engine": voice.engine.value, "model_id": getattr(md, "model_id", None) or voice.model_id,
                     "model_revision": getattr(md, "model_revision", None) if md else voice.model_revision,
                     "generation_parameters": spec.generation_parameters, "seed": spec.seed},
                    wav.size / engine.sample_rate, problems, audition_id)
            finally:
                engine.close()
            with self._lock:
                entry.update(status="done", finished_at=_now(), problems=problems)
        except (PipelineError, TTSEngineError, ConflictError, NotFoundError, OSError, ValueError) as exc:
            log.error("Audition %s failed: %s", audition_id, exc)
            with self._lock:
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=_now())

    # ------------------------------------------------------------------ worker
    def _schedule(self, job_id: str) -> None:
        self._executor.submit(self._run, job_id)

    def _run(self, job_id: str) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None or job_id in self._active or rec.state not in (JobState.QUEUED, JobState.GENERATING):
                return  # cancelled while queued, or already being processed (duplicate submission)
            self._active.add(job_id)
            cancel = self._cancel.setdefault(job_id, threading.Event())
            cancel.clear()
        token = current_job_id.set(job_id)
        out_dir = self.job_output_dir(job_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(out_dir / "job.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.addFilter(_JobLogFilter(job_id))
        logging.getLogger().addHandler(handler)
        try:
            if rec.phase == "generation":
                self._run_generation(job_id, out_dir, cancel)
            else:
                self._run_analysis(job_id, out_dir, cancel)
        except JobCancelled:
            log.warning("Job %s cancelled", job_id)
            self._close_stage(job_id, "cancelled")
            self._update(job_id, state=JobState.CANCELLED, finished_at=_now(), message="Cancelled")
        except Exception as exc:  # the worker must record every failure, whatever its type
            log.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
            self._close_stage(job_id, "failed")
            self._update(job_id, state=JobState.FAILED, finished_at=_now(),
                         error=f"{type(exc).__name__}: {exc}", message="Failed")
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()
            current_job_id.reset(token)
            with self._lock:
                self._active.discard(job_id)

    def _run_analysis(self, job_id: str, out_dir: Path, cancel: threading.Event) -> None:
        self._update(job_id, state=JobState.RUNNING, started_at=_now(), finished_at=None, error=None,
                     stage="validate", progress=0.0, message="Starting", phase="analysis")
        rec = self.get(job_id)
        opts = JobOptions(**rec.options.model_dump())
        cb = self._progress_cb(job_id, cancel, "analysis")
        if not rec.review_required:
            result = self.pipeline.run(Path(rec.input_path), opts, out_dir, cb)
            self._finish(job_id, result)
            return
        analysis = self.pipeline.analyze_for_review(Path(rec.input_path), opts, out_dir, cb)
        if cancel.is_set():
            raise JobCancelled()
        try:  # a retried/restarted analysis must never discard edits already saved for this job
            draft = self.load_draft(job_id)
            log.info("Keeping existing draft v%d for job %s", draft.version, job_id)
        except NotFoundError:
            draft = dr.build_draft(job_id=job_id, project_id=rec.project_id, video_id=rec.video_id,
                                   analysis=analysis, mix_mode=opts.mix_mode, settings=self.s, bank=self.pipeline.bank)
        try:
            draft = self.suggest_identities(job_id, draft, rec.project_id)
        except Exception as exc:  # suggestions are advisory; never fail analysis because of them
            log.warning("Identity suggestions failed for job %s: %s", job_id, exc)
        self.projects.save_draft(draft)
        self._close_stage(job_id, "completed")
        self._update(job_id, state=JobState.AWAITING_REVIEW, stage="awaiting_review", progress=1.0,
                     message="Draft ready for review", source_language=analysis.source_language,
                     speakers=len(draft.speakers), segments=len(draft.segments), warnings=draft.warnings,
                     analysis_completed_at=_now())
        log.info("Job %s analysed: %d segments, %d speakers; awaiting review", job_id, len(draft.segments),
                 len(draft.speakers))

    def _run_generation(self, job_id: str, out_dir: Path, cancel: threading.Event) -> None:
        rec = self.get(job_id)
        self._update(job_id, state=JobState.GENERATING, started_at=rec.started_at or _now(), finished_at=None,
                     error=None, stage="prepare", progress=0.0, message="Starting generation")
        approved = self.projects.load_approved(rec.project_id, rec.video_id, rec.approved_version)
        index = self.analysis_index(job_id)
        blocks, translations, requests, locked, label_map = dr.to_generation_inputs(approved)
        turns = [{**t, "speaker": label_map.get(t["speaker"], t["speaker"])} for t in index["turns"]]
        plan = GenerationPlan(blocks=blocks, translations=translations, source_language=approved.source_language,
                              voice_requests=requests, locked=locked,
                              library=self.projects.library_lookup(rec.project_id),
                              voice_library=self.voice_library.lookup if self.voice_library else None,
                              characters=self.character_lookup(rec.project_id))
        opts = JobOptions(**rec.options.model_dump())
        result = self.pipeline.generate_reviewed(Path(rec.input_path), turns, plan, opts, out_dir,
                                                 self._progress_cb(job_id, cancel, "generation"))
        self._finish(job_id, result, extra_warnings=list(approved.warnings))
        self.projects.touch_project(rec.project_id)
        if self.series is not None:
            for spk, rv in result.resolved.items():
                if rv.character_id:
                    self.series.log_generation(rec.project_id, rv.character_id, {
                        "job_id": job_id, "video_id": rec.video_id, "local_speaker_id": spk,
                        "voice_profile_id": rv.voice_profile_id, "engine": rv.engine.value, "voice_id": rv.voice_id,
                        "approved_version": rec.approved_version, "qa_status": result.qa_status, "at": _now()})

    def _finish(self, job_id: str, result, extra_warnings: Optional[list[str]] = None) -> None:
        self._close_stage(job_id, "completed")
        warnings = list(dict.fromkeys((extra_warnings or []) + result.warnings))
        qa_status = getattr(result, "qa_status", None)
        self._update(job_id, state=JobState.SUCCEEDED, stage="done", progress=1.0, finished_at=_now(),
                     message="Dubbing complete" + (f" (QA {qa_status})" if qa_status else ""),
                     output_path=os.fspath(result.output_video),
                     manifest_path=os.fspath(result.manifest), timing_report_path=os.fspath(result.timing_report),
                     source_language=result.source_language, speakers=result.speakers,
                     segments=result.segments, warnings=warnings, qa_status=qa_status,
                     qa_report_path=os.fspath(result.qa_report) if getattr(result, "qa_report", None) else None,
                     release_approved=False,
                     engines=sorted({rv.engine.value for rv in getattr(result, "resolved", {}).values()}))
        log.info("Job %s succeeded -> %s", job_id, result.output_video)

    def _progress_cb(self, job_id: str, cancel: threading.Event, phase: str):
        last = {"pct": -1.0, "stage": None}

        def cb(stage: str, fraction: float, message: str) -> None:
            if cancel.is_set():
                raise JobCancelled()
            pct = round(fraction * 100, 1)
            if pct == last["pct"] and stage == last["stage"]:
                return
            if stage != last["stage"]:
                self._open_stage(job_id, stage, phase, message)
            last["pct"], last["stage"] = pct, stage
            self._update(job_id, stage=stage, progress=round(fraction, 4), message=message)

        return cb

    def _open_stage(self, job_id: str, stage: str, phase: str, message: str) -> None:
        with self._lock:
            rec = self._jobs[job_id]
            history = [e.model_copy() for e in rec.stage_history]
            for e in history:
                if e.status == "running":
                    e.status, e.finished_at = "completed", _now()
            existing = next((e for e in history if e.stage == stage and e.phase == phase), None)
            if existing:  # a stage reported again (e.g. retry) restarts its entry
                existing.status, existing.started_at, existing.finished_at, existing.message = "running", _now(), None, message
            else:
                history.append(StageEvent(stage=stage, phase=phase, started_at=_now(), message=message))
            self._update(job_id, stage_history=history)

    def _close_stage(self, job_id: str, status: str) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return
            history = [e.model_copy() for e in rec.stage_history]
            for e in history:
                if e.status == "running":
                    e.status, e.finished_at = status, _now()
            self._update(job_id, stage_history=history)

    def _profiles(self, job_id: str) -> dict:
        index = self.analysis_index(job_id)
        f = Path(index["voices_dir"]) / "profiles.json"
        return read_json(f) if f.is_file() else {}

    # ------------------------------------------------------------------ persistence
    @staticmethod
    def _check_transition(old: JobState, new: JobState) -> None:
        if new not in _TRANSITIONS[old]:
            raise InvalidTransition(f"Cannot move a {old.value} job to {new.value}.")

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            rec = self._jobs[job_id]
            new_state = changes.get("state")
            if new_state is not None and new_state != rec.state:
                self._check_transition(rec.state, new_state)
            updated = rec.model_copy(update=changes)
            self._jobs[job_id] = updated
            self._persist(updated)

    def _persist(self, rec: JobRecord) -> None:
        path = self.s.jobs_dir / f"{rec.job_id}.json"
        tmp = path.with_name(f"{path.name}.part")
        tmp.write_text(rec.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _load_and_resume(self) -> None:
        resume = []
        for path in sorted(self.s.jobs_dir.glob("*.json")):
            try:
                rec = JobRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, OSError) as exc:
                log.warning("Ignoring unreadable job record %s: %s", path.name, exc)
                continue
            self._jobs[rec.job_id] = rec
            if rec.state in ACTIVE:
                if Path(rec.input_path).is_file():
                    resume.append(rec)
                else:
                    self._update(rec.job_id, state=JobState.FAILED, finished_at=_now(),
                                 error="Input file missing after server restart.")
        for rec in resume:
            if rec.state == JobState.RUNNING:
                self._update(rec.job_id, state=JobState.QUEUED, message="Re-queued after server restart")
            elif rec.state == JobState.GENERATING:
                self._update(rec.job_id, message="Generation re-queued after server restart")
            self._schedule(rec.job_id)
            log.info("Job %s re-queued after restart", rec.job_id)


def _now() -> datetime:
    return datetime.now(timezone.utc)
