# hindi-dubbing-platform/backend/app/api.py
"""Project, review, Speaker Library and artifact endpoints (``/api/v1``).

The original single-shot routes stay in ``main.py`` unchanged. Everything here only reads or writes
records and streams files; model inference and long FFmpeg work always run on the job workers.
Local filesystem paths are never accepted from clients; ids are validated before use.
"""

from __future__ import annotations

import io
import logging
import mimetypes
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from backend.app.config import (
    ALLOWED_REFERENCE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    SUPPORTED_SOURCE_LANGUAGES,
    SUPPORTED_TARGET_LANGUAGES,
    TARGET_LANGUAGE,
    MixMode,
    TranslationEngine,
)
from backend.app.jobs import JobManager, JobOptionsModel, JobRecord, JobState
from backend.services import drafts as dr
from backend.services.media import MediaError
from backend.services.projects import (
    Project,
    ProjectSettings,
    ProjectSpeaker,
    ProjectStore,
    ProjectVideo,
    SpeakerSource,
)
from backend.services.storage import (
    SEGMENT_ID_RE,
    SPEAKER_LABEL_RE,
    ConflictError,
    NotFoundError,
    ValidationFailed,
    now,
    read_json,
    require_entity_id,
)
from backend.services.voices import VoiceSource

log = logging.getLogger("dubbing.api")
router = APIRouter(prefix="/api/v1")
_CHUNK = 4 * 1024 * 1024


# =============================================================================== helpers
def _jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def _store(request: Request) -> ProjectStore:
    return request.app.state.projects


def _job(request: Request, job_id: str) -> JobRecord:
    rec = _jobs(request).get(job_id)
    if rec is None:
        raise HTTPException(404, "Job not found")
    return rec


async def save_upload(file: UploadFile, dst: Path, limit_mb: int) -> int:
    limit = limit_mb * 1024 * 1024
    written = 0
    try:
        with open(dst, "wb") as f:
            while chunk := await file.read(_CHUNK):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"Upload exceeds the {limit_mb} MB limit")
                f.write(chunk)
        if written == 0:
            raise HTTPException(422, "Uploaded file is empty")
    finally:
        await file.close()
    return written


def _stream_file(request: Request, path: Path, media_type: Optional[str] = None,
                 download_name: Optional[str] = None) -> Response:
    """File response with HTTP Range support (needed for seeking in <video>/<audio>)."""
    if not path.is_file():
        raise HTTPException(410, "File no longer exists on disk")
    size = path.stat().st_size
    media_type = media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
    if download_name:
        headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
    rng = request.headers.get("range")
    if not rng or not rng.startswith("bytes="):
        return FileResponse(os.fspath(path), media_type=media_type, headers=headers)
    try:
        first = rng[6:].split(",")[0].strip()
        a, _, b = first.partition("-")
        if a:
            start = int(a)
            end = min(int(b), size - 1) if b else size - 1
        else:
            start, end = max(0, size - int(b)), size - 1
        if start > end or start >= size:
            raise ValueError
    except ValueError:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def body() -> Iterator[bytes]:
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                data = f.read(min(1 << 20, left))
                if not data:
                    break
                left -= len(data)
                yield data

    headers.update({"Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(end - start + 1)})
    return StreamingResponse(body(), status_code=206, media_type=media_type, headers=headers)


def _wav_bytes(path: Path, start: float, end: float, pad: float = 0.0) -> Response:
    info = sf.info(os.fspath(path))
    s0 = max(0, int((start - pad) * info.samplerate))
    s1 = min(info.frames, int((end + pad) * info.samplerate))
    data, sr = sf.read(os.fspath(path), start=s0, stop=max(s0 + 1, s1), dtype="float32", always_2d=True)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav", headers={"Cache-Control": "no-cache"})


def _options(project: Project, settings, source_language: Optional[str], mix_mode: Optional[MixMode],
             translation_engine: Optional[TranslationEngine], min_speakers: Optional[int],
             max_speakers: Optional[int]) -> JobOptionsModel:
    if source_language is not None:
        source_language = source_language.strip().lower() or None
    source_language = source_language or project.settings.source_language
    if source_language and source_language not in SUPPORTED_SOURCE_LANGUAGES:
        raise HTTPException(422, f"source_language must be one of {sorted(SUPPORTED_SOURCE_LANGUAGES)} or omitted")
    if min_speakers and max_speakers and min_speakers > max_speakers:
        raise HTTPException(422, "min_speakers cannot exceed max_speakers")
    engine = translation_engine or project.settings.translation_engine or settings.translation_engine
    problems = settings.preflight_errors(engine)
    if problems:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, {"configuration_problems": problems})
    return JobOptionsModel(source_language=source_language, mix_mode=mix_mode or project.settings.mix_mode,
                           translation_engine=engine, min_speakers=min_speakers, max_speakers=max_speakers)


# =============================================================================== meta
@router.get("/meta")
def meta(request: Request) -> dict:
    s = request.app.state.settings
    return {
        "source_languages": SUPPORTED_SOURCE_LANGUAGES, "target_languages": SUPPORTED_TARGET_LANGUAGES,
        "default_target_language": TARGET_LANGUAGE, "mix_modes": [m.value for m in MixMode],
        "default_mix_mode": s.default_mix_mode.value, "translation_engines": [e.value for e in TranslationEngine],
        "default_translation_engine": s.translation_engine.value, "review_enabled": s.project_review_enabled,
        "min_speaker_reference_seconds": s.min_speaker_reference_seconds,
        "max_upload_mb": s.max_upload_mb, "max_reference_upload_mb": s.max_reference_upload_mb,
        "video_extensions": sorted(ALLOWED_VIDEO_EXTENSIONS), "reference_extensions": sorted(ALLOWED_REFERENCE_EXTENSIONS),
        "configuration_problems": s.preflight_errors(s.translation_engine),
        # v3
        "voice_sources": [v.value for v in VoiceSource], "tts_engines_enabled": sorted(e.value for e in s.enabled_tts_engines),
        "separation_fallback": s.separation_fallback.value, "speaker_identity_enabled": s.speaker_identity_enabled,
        "identity_thresholds": {"high": s.speaker_match_high_threshold, "review": s.speaker_match_review_threshold,
                                "min_segment_identity_duration": s.min_segment_identity_duration},
        "mix": {"dynamic_ducking": s.dynamic_ducking_enabled, "music_duck_db": s.music_duck_db,
                "dialogue_target_lufs": s.dialogue_target_lufs, "true_peak_ceiling_dbtp": s.true_peak_ceiling_dbtp},
    }


# =============================================================================== projects
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field("", max_length=2000)
    target_language: str = TARGET_LANGUAGE
    settings: ProjectSettings = Field(default_factory=ProjectSettings)
    kind: str = Field("standalone", pattern="^(standalone|series)$")


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=2000)
    settings: Optional[ProjectSettings] = None
    status: Optional[str] = Field(None, pattern="^(active|archived)$")


class ProjectSummary(Project):
    video_count: int = 0
    speaker_count: int = 0
    jobs: dict[str, int] = Field(default_factory=dict)
    last_activity_at: Optional[datetime] = None


def _summary(request: Request, p: Project) -> ProjectSummary:
    store, jobs = _store(request), _jobs(request)
    recs = jobs.list(p.project_id)
    last = max([p.updated_at] + [r.finished_at or r.started_at or r.created_at for r in recs])
    return ProjectSummary(**p.model_dump(), video_count=len(store.list_videos(p.project_id)),
                          speaker_count=len(store.list_speakers(p.project_id)),
                          jobs=jobs.counts(p.project_id), last_activity_at=last)


def _check_target(lang: str) -> None:
    if lang not in SUPPORTED_TARGET_LANGUAGES:
        raise HTTPException(422, f"target_language must be one of {sorted(SUPPORTED_TARGET_LANGUAGES)}; "
                                 "other targets need validated translation + TTS models.")


@router.post("/projects", status_code=201, response_model=ProjectSummary)
def create_project(body: ProjectCreate, request: Request) -> ProjectSummary:
    _check_target(body.target_language)
    if body.settings.source_language and body.settings.source_language not in SUPPORTED_SOURCE_LANGUAGES:
        raise HTTPException(422, f"settings.source_language must be one of {sorted(SUPPORTED_SOURCE_LANGUAGES)}")
    p = _store(request).create_project(body.name, body.description, body.target_language, body.settings, body.kind)
    return _summary(request, p)


@router.get("/projects", response_model=list[ProjectSummary])
def list_projects(request: Request, q: Optional[str] = Query(None, max_length=120)) -> list[ProjectSummary]:
    items = _store(request).list_projects()
    if q:
        items = [p for p in items if q.casefold() in p.name.casefold() or q.casefold() in p.description.casefold()]
    return [_summary(request, p) for p in items]


class VideoWithJob(ProjectVideo):
    job: Optional[JobRecord] = None


class ProjectDetail(ProjectSummary):
    videos: list[VideoWithJob] = Field(default_factory=list)
    speakers: list[ProjectSpeaker] = Field(default_factory=list)


def _video_with_job(request: Request, v: ProjectVideo) -> VideoWithJob:
    return VideoWithJob(**v.model_dump(), job=_jobs(request).get(v.current_job_id) if v.current_job_id else None)


@router.get("/projects/{project_id}", response_model=ProjectDetail)
def get_project(project_id: str, request: Request) -> ProjectDetail:
    store = _store(request)
    p = store.get_project(project_id)
    return ProjectDetail(**_summary(request, p).model_dump(),
                         videos=[_video_with_job(request, v) for v in store.list_videos(project_id)],
                         speakers=store.list_speakers(project_id))


@router.patch("/projects/{project_id}", response_model=ProjectSummary)
def update_project(project_id: str, body: ProjectUpdate, request: Request) -> ProjectSummary:
    p = _store(request).update_project(project_id, **body.model_dump(exclude_unset=True))
    return _summary(request, p)


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: str, request: Request, force: bool = False) -> Response:
    store, jobs = _store(request), _jobs(request)
    store.get_project(project_id)
    recs = jobs.list(project_id)
    busy = [r.job_id for r in recs if r.state in (JobState.QUEUED, JobState.RUNNING, JobState.GENERATING)]
    if busy:
        raise ConflictError(f"Project has {len(busy)} job(s) in progress; cancel them first.")
    if (store.list_videos(project_id) or store.list_speakers(project_id)) and not force:
        raise ConflictError("Project contains videos or library speakers; pass force=true to delete everything.")
    store.delete_project(project_id)
    jobs.delete_records([r.job_id for r in recs])
    return Response(status_code=204)


# =============================================================================== project videos
class VideoUploadResponse(BaseModel):
    project_id: str
    video_id: str
    job_id: str
    status: JobState
    video: ProjectVideo


@router.post("/projects/{project_id}/videos", status_code=202, response_model=VideoUploadResponse)
async def upload_video(
    project_id: str,
    request: Request,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None, max_length=200),
    source_language: Optional[str] = Form(None),
    mix_mode: Optional[MixMode] = Form(None),
    translation_engine: Optional[TranslationEngine] = Form(None),
    min_speakers: Optional[int] = Form(None, ge=1, le=20),
    max_speakers: Optional[int] = Form(None, ge=1, le=20),
    episode_number: Optional[int] = Form(None, ge=0, le=10000),
) -> VideoUploadResponse:
    s, store, jobs = request.app.state.settings, _store(request), _jobs(request)
    project = store.get_project(project_id)
    options = _options(project, s, source_language, mix_mode, translation_engine, min_speakers, max_speakers)
    original_name = Path(file.filename or "").name
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}")
    video_id, dst = store.new_video_slot(project_id, ext)
    try:
        size = await save_upload(file, dst, s.max_upload_mb)
        info = await run_in_threadpool(request.app.state.ffmpeg.probe, dst)
    except MediaError as exc:
        store.discard_video_slot(project_id, video_id)
        raise HTTPException(422, f"Invalid media: {exc}") from exc
    except BaseException:
        store.discard_video_slot(project_id, video_id)
        raise
    ts = now()
    video = store.add_video(ProjectVideo(
        video_id=video_id, project_id=project_id, title=(title or Path(original_name).stem).strip()[:200],
        original_filename=original_name, stored_filename=dst.name, size_bytes=size, duration=info.duration,
        width=info.width, height=info.height, fps=info.fps, episode_number=episode_number, created_at=ts, updated_at=ts))
    review = project.settings.review_enabled if project.settings.review_enabled is not None else s.project_review_enabled
    job_id = jobs.new_job_id()
    video = store.attach_job(project_id, video_id, job_id)
    rec = jobs.submit(dst, original_name, options, job_id, project_id=project_id, video_id=video_id,
                      review_required=review)
    return VideoUploadResponse(project_id=project_id, video_id=video_id, job_id=job_id, status=rec.state, video=video)


@router.get("/projects/{project_id}/videos", response_model=list[VideoWithJob])
def list_videos(project_id: str, request: Request) -> list[VideoWithJob]:
    return [_video_with_job(request, v) for v in _store(request).list_videos(project_id)]


@router.get("/projects/{project_id}/videos/{video_id}", response_model=VideoWithJob)
def get_video(project_id: str, video_id: str, request: Request) -> VideoWithJob:
    return _video_with_job(request, _store(request).get_video(project_id, video_id))


# =============================================================================== jobs (additional)
@router.get("/jobs", response_model=list[JobRecord])
def list_jobs(request: Request, project_id: Optional[str] = None, limit: int = Query(100, ge=1, le=1000)) -> list[JobRecord]:
    return _jobs(request).list(project_id)[:limit]


@router.post("/jobs/{job_id}/cancel", response_model=JobRecord)
def cancel_job(job_id: str, request: Request) -> JobRecord:
    return _jobs(request).cancel(job_id)


@router.post("/jobs/{job_id}/retry", response_model=JobRecord)
def retry_job(job_id: str, request: Request) -> JobRecord:
    return _jobs(request).retry(job_id)


@router.get("/jobs/{job_id}/logs")
def job_logs(job_id: str, request: Request, tail: int = Query(200, ge=1, le=5000)) -> dict:
    _job(request, job_id)
    f = _jobs(request).job_output_dir(job_id) / "job.log"
    lines = f.read_text(encoding="utf-8", errors="replace").splitlines() if f.is_file() else []
    return {"job_id": job_id, "lines": lines[-tail:], "total_lines": len(lines)}


def _job_json(request: Request, job_id: str, name: str) -> dict:
    _job(request, job_id)
    f = _jobs(request).job_output_dir(job_id) / name
    if not f.is_file():
        raise HTTPException(404, f"{name} is not available yet")
    return read_json(f)


@router.get("/jobs/{job_id}/timing-report")
def timing_report(job_id: str, request: Request) -> dict:
    return _job_json(request, job_id, "timing_report.json")


@router.get("/jobs/{job_id}/voice-resolution")
def voice_resolution(job_id: str, request: Request) -> dict:
    return _job_json(request, job_id, "voice_resolution.json")


@router.get("/jobs/{job_id}/report")
def job_report(job_id: str, request: Request) -> dict:
    return _job_json(request, job_id, "job_report.json")


@router.get("/jobs/{job_id}/manifest")
def manifest(job_id: str, request: Request, download: bool = False) -> Response:
    rec = _job(request, job_id)
    if rec.manifest_path and Path(rec.manifest_path).is_file():
        path = Path(rec.manifest_path)
    elif rec.project_id and rec.video_id:
        jobs = _jobs(request)
        jobs.load_draft(job_id)
        path = _store(request).drafts_dir(rec.project_id, rec.video_id) / "draft.json"
    else:
        raise HTTPException(404, "No manifest yet")
    return FileResponse(os.fspath(path), media_type="application/json",
                        filename=f"{job_id}.{path.stem}.json" if download else None)


@router.get("/jobs/{job_id}/source-video")
def source_video(job_id: str, request: Request) -> Response:
    rec = _job(request, job_id)
    return _stream_file(request, Path(rec.input_path))


@router.get("/jobs/{job_id}/output-video")
def output_video(job_id: str, request: Request) -> Response:
    rec = _job(request, job_id)
    if rec.state != JobState.SUCCEEDED or not rec.output_path:
        raise HTTPException(409, f"Job is {rec.state.value}; output not available")
    return _stream_file(request, Path(rec.output_path))


@router.get("/jobs/{job_id}/segments/{segment_id}/original-audio")
def original_segment_audio(job_id: str, segment_id: str, request: Request,
                           stem: str = Query("master", pattern="^(master|vocals)$")) -> Response:
    _job(request, job_id)
    if not SEGMENT_ID_RE.fullmatch(segment_id):
        raise HTTPException(404, "Segment not found")
    jobs = _jobs(request)
    index = jobs.analysis_index(job_id)
    try:
        seg = next(s for s in jobs.load_draft(job_id).segments if s.segment_id == segment_id)
        start, end = seg.start, seg.end
    except (NotFoundError, StopIteration):
        rec = _job(request, job_id)
        segs = read_json(Path(rec.manifest_path)) if rec.manifest_path else []
        hit = next((s for s in segs if s["segment_id"] == segment_id), None)
        if hit is None:
            raise HTTPException(404, "Segment not found") from None
        start, end = hit["start_time"], hit["end_time"]
    return _wav_bytes(Path(index[stem]), start, end, pad=0.15)


@router.get("/jobs/{job_id}/segments/{segment_id}/dub-audio")
def dub_segment_audio(job_id: str, segment_id: str, request: Request) -> Response:
    rec = _job(request, job_id)
    if rec.state != JobState.SUCCEEDED or not rec.manifest_path:
        raise HTTPException(409, "Dubbed audio exists only after generation")
    seg = next((s for s in read_json(Path(rec.manifest_path)) if s["segment_id"] == segment_id), None)
    if seg is None or not seg.get("generated_audio_path"):
        raise HTTPException(404, "Segment not found")
    return _stream_file(request, Path(seg["generated_audio_path"]), "audio/wav")


def _clone_reference(request: Request, job_id: str, speaker_id: str) -> tuple[Path, float]:
    """Clone reference of a speaker as built during analysis (diarization label or merged speaker)."""
    if not SPEAKER_LABEL_RE.fullmatch(speaker_id):
        raise HTTPException(404, "Speaker not found")
    jobs = _jobs(request)
    profiles = jobs._profiles(job_id)
    labels = [speaker_id]
    try:
        spk = next((s for s in jobs.load_draft(job_id).speakers if s.speaker_id == speaker_id), None)
        if spk:
            labels = spk.origin_labels or labels
    except NotFoundError:
        pass
    best = max((profiles.get(lbl) or {} for lbl in labels), key=lambda p: p.get("seconds", 0.0), default={})
    if not best.get("reference"):
        raise HTTPException(404, "No clone reference is available for this speaker")
    return Path(jobs.analysis_index(job_id)["voices_dir"]) / best["reference"], float(best.get("seconds", 0.0))


@router.get("/jobs/{job_id}/speakers/{speaker_id}/reference-audio")
def job_speaker_reference(job_id: str, speaker_id: str, request: Request) -> Response:
    _job(request, job_id)
    path, _ = _clone_reference(request, job_id, speaker_id)
    return _stream_file(request, path, "audio/wav")


# =============================================================================== draft review
@router.get("/jobs/{job_id}/draft", response_model=dr.DraftManifest)
def get_draft(job_id: str, request: Request) -> dr.DraftManifest:
    _job(request, job_id)
    return dr.present(_jobs(request).load_draft(job_id))


@router.patch("/jobs/{job_id}/draft", response_model=dr.DraftManifest)
def patch_draft(job_id: str, body: dr.DraftPatch, request: Request) -> dr.DraftManifest:
    rec = _job(request, job_id)
    library = _store(request).list_speakers(rec.project_id) if rec.project_id else []
    lib_ids = {s.speaker_id for s in library}
    for sp in body.speakers:
        if sp.library_speaker_id and sp.library_speaker_id not in lib_ids:
            raise ValidationFailed("Draft update rejected",
                                   [f"{sp.speaker_id}: library speaker does not belong to this project"])
    return dr.present(_jobs(request).update_draft(job_id, body))


class ConfirmRequest(BaseModel):
    base_version: int = Field(..., ge=1)


class ConfirmResponse(BaseModel):
    job: JobRecord
    approved_version: int
    warnings: list[str]


@router.post("/jobs/{job_id}/confirm", status_code=202, response_model=ConfirmResponse)
def confirm(job_id: str, body: ConfirmRequest, request: Request) -> ConfirmResponse:
    _job(request, job_id)
    rec, warnings = _jobs(request).confirm(job_id, body.base_version)
    return ConfirmResponse(job=rec, approved_version=rec.approved_version, warnings=warnings)


@router.get("/jobs/{job_id}/confirm-preview")
def confirm_preview(job_id: str, request: Request) -> dict:
    """Dry-run validation used by the UI's confirmation summary."""
    rec = _job(request, job_id)
    jobs = _jobs(request)
    draft = jobs.load_draft(job_id)
    errors, warnings = dr.validate_for_confirm(draft, _store(request).library_ids_with_transcript(rec.project_id),
                                               jobs.pipeline.bank, request.app.state.settings,
                                               cast_ids=jobs.cast_ids(), character_ids=jobs.characters_with_voice(rec.project_id),
                                               profiles=jobs._profiles(job_id))
    return {"version": draft.version, "errors": errors, "warnings": warnings, "speakers": [
        {"speaker_id": s.speaker_id, "display_name": s.display_name, "voice_source": s.voice_source,
         "library_speaker_id": s.library_speaker_id, "default_voice_id": s.default_voice_id,
         "engine": s.engine, "voice_id": s.voice_id, "character_id": s.character_id, "character_name": s.character_name,
         "mapping_status": s.mapping_status, "segment_count": s.segment_count} for s in draft.speakers if s.segment_count]}


# =============================================================================== default voices
@router.get("/default-voices")
def default_voices(request: Request) -> dict:
    bank = _jobs(request).pipeline.bank
    return {"directory_configured": True, "voices": [v.public() for v in bank.list()],
            "invalid": [{"voice_id": v.voice_id, "problem": v.problem} for v in bank.list(include_invalid=True)
                        if not v.valid]}


@router.get("/default-voices/{voice_id}/audio")
def default_voice_audio(voice_id: str, request: Request) -> Response:
    v = _jobs(request).pipeline.bank.get(voice_id)
    if v is None:
        raise HTTPException(404, "Default voice not found")
    return _stream_file(request, v.path, "audio/wav")


# =============================================================================== speaker library
class SpeakerCreate(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=80)
    notes: str = Field("", max_length=2000)


class SpeakerUpdate(BaseModel):
    display_name: Optional[str] = Field(None, min_length=1, max_length=80)
    notes: Optional[str] = Field(None, max_length=2000)
    reference_transcript: Optional[str] = Field(None, max_length=4000)


class PromoteRequest(BaseModel):
    job_id: str
    job_speaker_id: str
    display_name: Optional[str] = Field(None, min_length=1, max_length=80)
    target_speaker_id: Optional[str] = None  # replace the reference of an existing library speaker
    replace: bool = False
    reference_transcript: Optional[str] = Field(None, max_length=4000)


@router.get("/projects/{project_id}/speakers", response_model=list[ProjectSpeaker])
def list_speakers(project_id: str, request: Request) -> list[ProjectSpeaker]:
    _store(request).get_project(project_id)
    return _store(request).list_speakers(project_id)


@router.post("/projects/{project_id}/speakers", status_code=201, response_model=ProjectSpeaker)
def create_speaker(project_id: str, body: SpeakerCreate, request: Request) -> ProjectSpeaker:
    return _store(request).create_speaker(project_id, body.display_name, body.notes)


@router.get("/projects/{project_id}/speakers/{speaker_id}", response_model=ProjectSpeaker)
def get_speaker(project_id: str, speaker_id: str, request: Request) -> ProjectSpeaker:
    return _store(request).get_speaker(project_id, speaker_id)


@router.patch("/projects/{project_id}/speakers/{speaker_id}", response_model=ProjectSpeaker)
def update_speaker(project_id: str, speaker_id: str, body: SpeakerUpdate, request: Request) -> ProjectSpeaker:
    return _store(request).update_speaker(project_id, speaker_id, body.display_name, body.notes,
                                          body.reference_transcript)


@router.delete("/projects/{project_id}/speakers/{speaker_id}", status_code=204)
def delete_speaker(project_id: str, speaker_id: str, request: Request) -> Response:
    store, jobs = _store(request), _jobs(request)
    store.get_speaker(project_id, speaker_id)
    users = jobs.jobs_using_library_speaker(project_id, speaker_id)
    if users:
        raise ConflictError(f"Speaker is selected in {len(users)} open job(s) ({', '.join(u[:8] for u in users)}); "
                            "change their voice selection first.")
    store.delete_speaker(project_id, speaker_id)
    return Response(status_code=204)


@router.post("/projects/{project_id}/speakers/{speaker_id}/reference", response_model=ProjectSpeaker)
async def upload_reference(project_id: str, speaker_id: str, request: Request, file: UploadFile = File(...),
                           replace: bool = Form(False),
                           transcript: Optional[str] = Form(None, max_length=4000)) -> ProjectSpeaker:
    s, store = request.app.state.settings, _store(request)
    store.get_speaker(project_id, speaker_id)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_REFERENCE_EXTENSIONS:
        raise HTTPException(415, f"Unsupported audio type '{ext}'. Allowed: {sorted(ALLOWED_REFERENCE_EXTENSIONS)}")
    tmpdir = Path(tempfile.mkdtemp(prefix="ref-", dir=store.speaker_dir(project_id, speaker_id)))
    try:
        src = tmpdir / f"upload{ext}"
        await save_upload(file, src, s.max_reference_upload_mb)
        spk, _ = await run_in_threadpool(store.set_reference, project_id, speaker_id, src, SpeakerSource.UPLOADED,
                                         replace, {"uploaded_filename": Path(file.filename or "").name}, transcript)
        return spk
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@router.get("/projects/{project_id}/speakers/{speaker_id}/reference")
def get_reference(project_id: str, speaker_id: str, request: Request) -> Response:
    store = _store(request)
    store.get_speaker(project_id, speaker_id)
    return _stream_file(request, store.reference_path(project_id, speaker_id), "audio/wav")


@router.post("/projects/{project_id}/speakers/promote", status_code=201, response_model=ProjectSpeaker)
async def promote_speaker(project_id: str, body: PromoteRequest, request: Request) -> ProjectSpeaker:
    """Save a voice used (or detected) in one of this project's jobs into the Speaker Library."""
    store, jobs = _store(request), _jobs(request)
    store.get_project(project_id)
    require_entity_id(body.job_id, "Job")
    rec = _job(request, body.job_id)
    if rec.project_id != project_id:
        raise HTTPException(404, "Job not found in this project")
    source, origin = SpeakerSource.CLONED, {"job_id": rec.job_id, "video_id": rec.video_id,
                                            "job_speaker_id": body.job_speaker_id}
    profiles = jobs._profiles(rec.job_id)
    try:
        draft_speaker = next(s for s in jobs.load_draft(rec.job_id).speakers
                             if s.speaker_id == body.job_speaker_id)
        labels = draft_speaker.origin_labels or [body.job_speaker_id]
    except (NotFoundError, StopIteration):
        labels = [body.job_speaker_id]
    transcript = body.reference_transcript or next(
        ((profiles.get(label) or {}).get("transcript") for label in labels
         if (profiles.get(label) or {}).get("transcript")), None)
    resolution_file = jobs.job_output_dir(rec.job_id) / "voice_resolution.json"
    resolved = read_json(resolution_file).get(body.job_speaker_id) if resolution_file.is_file() else None
    if resolved and resolved["resolved_source"] == VoiceSource.LIBRARY.value:
        raise ConflictError("This speaker already uses a library voice.")
    if resolved and resolved["resolved_source"] == VoiceSource.DEFAULT.value:
        voice = jobs.pipeline.bank.get(resolved["default_voice_id"])
        if voice is None:
            raise HTTPException(404, "The default voice used by this speaker is no longer available")
        src, source = voice.path, SpeakerSource.DEFAULT_PROMOTED
        origin["default_voice_id"] = voice.voice_id
    else:
        src, _ = _clone_reference(request, rec.job_id, body.job_speaker_id)
    if body.target_speaker_id:
        target = store.get_speaker(project_id, body.target_speaker_id)
    else:
        name = body.display_name
        if not name:
            try:
                name = next(s.display_name for s in jobs.load_draft(rec.job_id).speakers
                            if s.speaker_id == body.job_speaker_id)
            except (NotFoundError, StopIteration):
                name = body.job_speaker_id
        target = store.create_speaker(project_id, name)
    try:
        spk, _ = await run_in_threadpool(store.set_reference, project_id, target.speaker_id, src, source,
                                         body.replace, origin, transcript)
    except Exception:
        if not body.target_speaker_id:  # do not leave an empty speaker behind
            store.delete_speaker(project_id, target.speaker_id)
        raise
    return spk

