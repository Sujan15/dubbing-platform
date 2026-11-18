# hindi-dubbing-platform/backend/app/main.py

"""FastAPI entry points: submit a dubbing job, poll its status, download the dubbed video.

Project, review and Speaker Library routes live in ``backend.app.api``; the built web UI
(``frontend/dist``) is served from ``/`` when present.

Run from the repository root:  python -m backend.app.main
"""

from __future__ import annotations

import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    SUPPORTED_SOURCE_LANGUAGES,
    MixMode,
    TranslationEngine,
    get_settings,
)
from backend.app.api import router as project_router
from backend.app.api_v3 import router as v3_router
from backend.app.config import REPO_ROOT
from backend.app.jobs import JobManager, JobOptionsModel, JobRecord, JobState
from backend.services.media import FFmpeg, MediaError
from backend.services.projects import ProjectStore
from backend.services.series import SeriesStore
from backend.services.storage import ConflictError, NotFoundError, ValidationFailed
from backend.services.voice_library import VoiceLibrary
from backend.services.voices import ReferenceAudioError

log = logging.getLogger("dubbing.api")
_CHUNK = 4 * 1024 * 1024


def _configure_logging(level: str) -> None:
    root = logging.getLogger()
    if not any(getattr(h, "_dubbing", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        handler._dubbing = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "urllib3", "numba", "matplotlib", "speechbrain", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    settings.ensure_dirs()
    settings.apply_runtime_environment()
    ff = FFmpeg(settings.ffmpeg_bin, settings.ffprobe_bin, settings.ffmpeg_timeout_seconds)
    try:
        app.state.ffmpeg_version = ff.version()
    except MediaError as exc:
        log.error("%s", exc)
        raise
    app.state.settings = settings
    app.state.ffmpeg = ff
    app.state.projects = ProjectStore(settings, ff)
    app.state.voice_library = VoiceLibrary(settings, ff)
    app.state.series = SeriesStore(settings, settings.projects_dir)
    app.state.jobs = JobManager(settings, app.state.projects, app.state.voice_library, app.state.series)
    log.info("Ready on http://%s:%d  | data dir: %s | %s", settings.host, settings.port, settings.data_dir,
             app.state.ffmpeg_version)
    for problem in settings.preflight_errors(settings.translation_engine):
        log.warning("Configuration: %s", problem)
    try:
        yield
    finally:
        app.state.jobs.shutdown()


app = FastAPI(title="Hindi AI Dubbing Platform", version="3.0.0", lifespan=lifespan)
app.include_router(project_router)
app.include_router(v3_router)
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


@app.exception_handler(NotFoundError)
async def _not_found(request: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ConflictError)
async def _conflict(request: Request, exc: ConflictError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(ValidationFailed)
async def _invalid(request: Request, exc: ValidationFailed) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc), "errors": exc.details})


@app.exception_handler(ReferenceAudioError)
async def _bad_reference(request: Request, exc: ReferenceAudioError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/api/v1/health")
def health(request: Request) -> dict:
    s = request.app.state.settings
    return {
        "status": "ok",
        "ffmpeg": request.app.state.ffmpeg_version,
        "configured_device": s.device,
        "translation_engine": s.translation_engine.value,
        "configuration_problems": s.preflight_errors(s.translation_engine),
        "jobs": request.app.state.jobs.counts(),
    }


@app.post("/api/v1/jobs", status_code=status.HTTP_202_ACCEPTED, response_model=JobRecord)
async def create_job(
    request: Request,
    file: UploadFile = File(..., description="Source video (English, Mandarin or Korean dialogue)"),
    source_language: Optional[str] = Form(None, description="en | zh | ko; omit to auto-detect"),
    mix_mode: Optional[MixMode] = Form(None),
    translation_engine: Optional[TranslationEngine] = Form(None),
    min_speakers: Optional[int] = Form(None, ge=1, le=20),
    max_speakers: Optional[int] = Form(None, ge=1, le=20),
) -> JobRecord:
    s = request.app.state.settings
    manager: JobManager = request.app.state.jobs

    if source_language is not None:
        source_language = source_language.strip().lower() or None
    if source_language and source_language not in SUPPORTED_SOURCE_LANGUAGES:
        raise HTTPException(422, f"source_language must be one of {sorted(SUPPORTED_SOURCE_LANGUAGES)} or omitted")
    if min_speakers and max_speakers and min_speakers > max_speakers:
        raise HTTPException(422, "min_speakers cannot exceed max_speakers")
    engine = translation_engine or s.translation_engine
    problems = s.preflight_errors(engine)
    if problems:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, {"configuration_problems": problems})

    original_name = Path(file.filename or "").name
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}")

    job_id = manager.new_job_id()
    job_dir = s.uploads_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    dst = job_dir / f"input{ext}"
    limit = s.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with open(dst, "wb") as f:
            while chunk := await file.read(_CHUNK):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"Upload exceeds MAX_UPLOAD_MB={s.max_upload_mb}")
                f.write(chunk)
        if written == 0:
            raise HTTPException(422, "Uploaded file is empty")
        await run_in_threadpool(request.app.state.ffmpeg.probe, dst)  # reject non-video / no-audio early
    except MediaError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(422, f"Invalid media: {exc}") from exc
    except BaseException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    finally:
        await file.close()

    options = JobOptionsModel(source_language=source_language, mix_mode=mix_mode or s.default_mix_mode,
                              translation_engine=engine, min_speakers=min_speakers, max_speakers=max_speakers)
    return manager.submit(dst, original_name, options, job_id)


@app.get("/api/v1/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str, request: Request) -> JobRecord:
    record = request.app.state.jobs.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    return record


@app.get("/api/v1/jobs/{job_id}/result")
def get_result(job_id: str, request: Request) -> FileResponse:
    record = request.app.state.jobs.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    if record.state != JobState.SUCCEEDED or not record.output_path:
        raise HTTPException(409, f"Job is {record.state.value}; result not available")
    path = Path(record.output_path)
    if not path.is_file():
        raise HTTPException(410, "Output file no longer exists on disk")
    download_name = f"{Path(record.input_filename).stem}.hindi{path.suffix}"
    return FileResponse(os.fspath(path), filename=download_name, media_type="application/octet-stream")


# ---- Web UI (production build). Registered last so it never shadows /api or /docs. ------------
if (FRONTEND_DIST / "index.html").is_file():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(404, "Not found")
        candidate = (FRONTEND_DIST / full_path).resolve()
        if full_path and candidate.is_file() and FRONTEND_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")


def run() -> None:
    s = get_settings()
    uvicorn.run(app, host=s.host, port=s.port, log_level=s.log_level.lower(), workers=1)


if __name__ == "__main__":
    run()
