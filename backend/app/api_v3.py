# hindi-dubbing-platform/backend/app/api_v3.py

"""v3 endpoints (``/api/v1``, additive): TTS engines, the hybrid Voice Library with auditions, series
characters and permanent voice profiles, cross-episode identity, the QA gate and stem playback.

Every v2 route in ``api.py`` / ``main.py`` is unchanged. Handlers only read/write records and stream
files; auditions and generation run on the job workers.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.app.api import _job, _jobs, _store, _stream_file, save_upload
from backend.app.config import ALLOWED_REFERENCE_EXTENSIONS, TTSEngine
from backend.app.jobs import JobRecord, JobState
from backend.services.series import Character, SeriesStore
from backend.services.storage import ConflictError, NotFoundError, read_json, require_entity_id
from backend.services.tts import engine_status
from backend.services.voice_library import CATALOG_SLOT_RE, DEFAULT_AUDITION_TEXT, VoiceLibrary
from backend.services.voices import VoiceSource

log = logging.getLogger("dubbing.api.v3")
router = APIRouter(prefix="/api/v1")


def _lib(request: Request) -> VoiceLibrary:
    return request.app.state.voice_library


def _series(request: Request) -> SeriesStore:
    return request.app.state.series


# =============================================================================== engines
@router.get("/engines")
def engines(request: Request) -> dict:
    s = request.app.state.settings
    return {"engines": engine_status(s), "enabled": sorted(e.value for e in s.enabled_tts_engines),
            "default_engine": TTSEngine.PARLER.value, "default_clone_engine": TTSEngine.INDICF5.value}


# =============================================================================== voice library
class VoiceCreate(BaseModel):
    voice_name: str = Field(..., min_length=1, max_length=80)
    engine: TTSEngine = TTSEngine.PARLER
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$")
    description: str = Field("", max_length=2000)
    voice_description_prompt: Optional[str] = Field(None, max_length=2000)
    sample_text: Optional[str] = Field(None, max_length=500)
    generation_parameters: dict = Field(default_factory=dict)
    seed: Optional[int] = Field(None, ge=0)
    catalog_slot: Optional[str] = Field(None, pattern=CATALOG_SLOT_RE)
    reference_transcript: Optional[str] = Field(None, max_length=4000)


class VoiceUpdate(BaseModel):
    voice_name: Optional[str] = Field(None, min_length=1, max_length=80)
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$")
    description: Optional[str] = Field(None, max_length=2000)
    voice_description_prompt: Optional[str] = Field(None, max_length=2000)
    sample_text: Optional[str] = Field(None, max_length=500)
    generation_parameters: Optional[dict] = None
    seed: Optional[int] = Field(None, ge=0)
    reference_transcript: Optional[str] = Field(None, max_length=4000)
    active: Optional[bool] = None
    license_metadata: Optional[dict] = None


def _legacy_voices(request: Request, project_id: Optional[str]) -> list[dict]:
    """Expose historical references as read-only records without making XTTS selectable."""
    out = []
    for v in _jobs(request).pipeline.bank.list():
        out.append({"voice_id": v.voice_id, "voice_name": v.label, "kind": "legacy", "engine": "xtts",
                    "language": "hi", "gender": v.gender, "description": v.description, "approved": True,
                    "active": False, "selectable": False, "voice_source": VoiceSource.DEFAULT.value,
                    "reference_speech_seconds": v.speech_seconds, "model_id": None,
                    "license_metadata": {"status": "retired"},
                    "preview_url": f"/api/v1/default-voices/{v.voice_id}/audio", "legacy": True})
    if project_id:
        store = _store(request)
        for spk in store.list_speakers(project_id):
            usable = bool(spk.reference_available and spk.reference_transcript)
            out.append({"voice_id": spk.speaker_id, "voice_name": spk.display_name, "kind": "legacy",
                        "engine": "indicf5" if usable else "xtts",
                        "language": "hi", "gender": None, "description": spk.notes, "approved": spk.reference_available,
                        "active": usable, "selectable": False, "voice_source": VoiceSource.LIBRARY.value,
                        "reference_speech_seconds": spk.reference_speech_seconds, "project_id": project_id,
                        "model_id": request.app.state.settings.indicf5_model if usable else None,
                        "license_metadata": {"status": "historical_reference"},
                        "preview_url": f"/api/v1/projects/{project_id}/speakers/{spk.speaker_id}/reference",
                        "legacy": True})
    return out


@router.get("/voice-library")
def list_voices(request: Request, engine: Optional[TTSEngine] = None, kind: Optional[str] = Query(None, pattern="^(clone|cast|legacy)$"),
                gender: Optional[str] = None, approved: Optional[bool] = None, include_inactive: bool = True,
                include_legacy: bool = False, project_id: Optional[str] = None) -> dict:
    voices = [{**v.public(), "voice_source": VoiceSource.CAST.value, "legacy": False,
               "preview_url": f"/api/v1/voice-library/{v.voice_id}/preview" if v.preview else None}
              for v in _lib(request).list(engine, kind, gender, approved, include_inactive)]
    legacy = []
    if include_legacy and (engine in (None, TTSEngine.XTTS)) and kind in (None, "legacy"):
        legacy = [v for v in _legacy_voices(request, project_id)
                  if (gender is None or (v.get("gender") or "").casefold() == gender.casefold())
                  and (approved is None or v["approved"] == approved)]
    return {"voices": voices + legacy, "engines": engine_status(request.app.state.settings)}


@router.post("/voice-library", status_code=201)
def create_voice(body: VoiceCreate, request: Request) -> dict:
    s = request.app.state.settings
    if body.engine not in s.enabled_tts_engines:
        raise HTTPException(422, f"Engine '{body.engine.value}' is not enabled (TTS_ENGINES_ENABLED={s.tts_engines_enabled}).")
    v = _lib(request).create(voice_name=body.voice_name, engine=body.engine, gender=body.gender,
                             description=body.description, voice_description_prompt=body.voice_description_prompt,
                             sample_text=body.sample_text, generation_parameters=body.generation_parameters,
                             seed=body.seed, catalog_slot=body.catalog_slot, reference_transcript=body.reference_transcript,
                             origin={"created_via": "api"})
    return v.public()


@router.post("/voice-library/catalog/seed", status_code=201)
def seed_catalog(request: Request) -> dict:
    created, updated = _lib(request).seed_catalog()
    return {
        "created": [v.public() for v in created],
        "updated": [v.public() for v in updated],
        "note": "Catalog slots are inactive and unapproved until auditioned.",
    }


@router.get("/voice-library/{voice_id}")
def get_voice(voice_id: str, request: Request) -> dict:
    return _lib(request).get(voice_id).public()


@router.patch("/voice-library/{voice_id}")
def update_voice(voice_id: str, body: VoiceUpdate, request: Request) -> dict:
    return _lib(request).update(voice_id, **body.model_dump(exclude_unset=True)).public()


@router.delete("/voice-library/{voice_id}", status_code=204)
def delete_voice(voice_id: str, request: Request) -> Response:
    series, store = _series(request), _store(request)

    def in_use(vid: str) -> list[str]:
        hits = []
        for p in store.list_projects():
            hits += series.profiles_using(p.project_id, VoiceSource.CAST, vid)
            hits += series.profiles_using(p.project_id, VoiceSource.CLONE, vid)
        return hits
    _lib(request).delete(voice_id, in_use)
    return Response(status_code=204)


@router.post("/voice-library/{voice_id}/reference")
async def upload_voice_reference(voice_id: str, request: Request, file: UploadFile = File(...),
                                 transcript: Optional[str] = Form(None, max_length=4000),
                                 replace: bool = Form(False)) -> dict:
    s, lib = request.app.state.settings, _lib(request)
    lib.get(voice_id)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_REFERENCE_EXTENSIONS:
        raise HTTPException(415, f"Unsupported audio type '{ext}'. Allowed: {sorted(ALLOWED_REFERENCE_EXTENSIONS)}")
    tmpdir = Path(tempfile.mkdtemp(prefix="ref-", dir=lib.voice_dir(voice_id)))
    try:
        src = tmpdir / f"upload{ext}"
        await save_upload(file, src, s.max_reference_upload_mb)
        v, _ = await run_in_threadpool(lib.set_reference, voice_id, src, transcript, replace,
                                       {"uploaded_filename": Path(file.filename or "").name})
        return v.public()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class PromoteToLibrary(BaseModel):
    job_id: str
    job_speaker_id: str
    voice_name: str = Field(..., min_length=1, max_length=80)
    engine: TTSEngine = TTSEngine.INDICF5
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$")
    description: str = Field("", max_length=2000)
    reference_transcript: Optional[str] = Field(None, max_length=4000)


@router.post("/voice-library/promote", status_code=201)
async def promote_job_voice(body: PromoteToLibrary, request: Request) -> dict:
    """Save a transcript-bearing clone reference as an IndicF5 Voice Library profile."""
    from backend.app.api import _clone_reference

    jobs, lib = _jobs(request), _lib(request)
    require_entity_id(body.job_id, "Job")
    rec = _job(request, body.job_id)
    src, _ = _clone_reference(request, body.job_id, body.job_speaker_id)
    profiles = jobs._profiles(body.job_id)
    transcript = body.reference_transcript
    if not transcript:
        try:
            spk = next((s_ for s_ in jobs.load_draft(body.job_id).speakers if s_.speaker_id == body.job_speaker_id), None)
            labels = spk.origin_labels if spk else [body.job_speaker_id]
        except NotFoundError:
            labels = [body.job_speaker_id]
        transcript = next(((profiles.get(l) or {}).get("transcript") for l in labels if (profiles.get(l) or {}).get("transcript")), None)
    v = lib.create(voice_name=body.voice_name, engine=body.engine, gender=body.gender, description=body.description,
                   reference_transcript=transcript,
                   origin={"job_id": rec.job_id, "video_id": rec.video_id, "job_speaker_id": body.job_speaker_id})
    try:
        v, _ = await run_in_threadpool(lib.set_reference, v.voice_id, src, transcript, False, None)
    except Exception:
        lib.delete(v.voice_id, lambda _vid: [])
        raise
    return v.public()


class AuditionRequest(BaseModel):
    text: str = Field(DEFAULT_AUDITION_TEXT, min_length=1, max_length=500)


@router.post("/voice-library/{voice_id}/audition", status_code=202)
def audition(voice_id: str, body: AuditionRequest, request: Request) -> dict:
    return _jobs(request).start_audition(voice_id, body.text)


@router.get("/voice-library/auditions/{audition_id}")
def audition_status(audition_id: str, request: Request) -> dict:
    e = _jobs(request).get_audition(audition_id)
    if e is None:
        raise HTTPException(404, "Audition not found")
    return e


@router.get("/voice-library/{voice_id}/preview")
def voice_preview(voice_id: str, request: Request, run_id: Optional[str] = None) -> Response:
    lib = _lib(request)
    lib.get(voice_id)
    path = lib.audition_path(voice_id, run_id) if run_id else lib.preview_path(voice_id)
    return _stream_file(request, path, "audio/wav")


@router.get("/voice-library/{voice_id}/reference")
def voice_reference(voice_id: str, request: Request) -> Response:
    lib = _lib(request)
    lib.get(voice_id)
    return _stream_file(request, lib.reference_path(voice_id), "audio/wav")


class ApprovalRequest(BaseModel):
    approved: bool
    note: str = Field("", max_length=2000)
    quality_benchmark: Optional[dict] = None


@router.post("/voice-library/{voice_id}/approval")
def set_voice_approval(voice_id: str, body: ApprovalRequest, request: Request) -> dict:
    return _lib(request).set_approval(voice_id, body.approved, body.note, body.quality_benchmark).public()


@router.get("/voice-library/{voice_id}/runs")
def voice_runs(voice_id: str, request: Request, tail: int = Query(100, ge=1, le=2000)) -> dict:
    lib = _lib(request)
    lib.get(voice_id)
    f = lib.voice_dir(voice_id) / "runs.jsonl"
    lines = f.read_text(encoding="utf-8").splitlines() if f.is_file() else []
    import json
    return {"voice_id": voice_id, "runs": [json.loads(ln) for ln in lines[-tail:] if ln.strip()]}


# =============================================================================== characters
class CharacterCreate(BaseModel):
    canonical_name: str = Field(..., min_length=1, max_length=80)
    aliases: list[str] = Field(default_factory=list, max_length=20)
    description: str = Field("", max_length=2000)
    role: Optional[str] = Field(None, pattern="^(hero|heroine|villain|support|narrator|other)$")
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$")


class CharacterUpdate(BaseModel):
    canonical_name: Optional[str] = Field(None, min_length=1, max_length=80)
    aliases: Optional[list[str]] = Field(None, max_length=20)
    description: Optional[str] = Field(None, max_length=2000)
    role: Optional[str] = Field(None, pattern="^(hero|heroine|villain|support|narrator|other)$")
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$")
    status: Optional[str] = Field(None, pattern="^(active|archived)$")


class VoiceProfileCreate(BaseModel):
    voice_source: VoiceSource
    voice_asset_id: Optional[str] = None
    label: str = Field("", max_length=120)
    notes: str = Field("", max_length=2000)
    make_active: bool = True
    approved: bool = True


class VoiceProfileUpdate(BaseModel):
    label: Optional[str] = Field(None, max_length=120)
    notes: Optional[str] = Field(None, max_length=2000)
    identity_locked: Optional[bool] = None
    approved: Optional[bool] = None
    make_active: Optional[bool] = None


def _character_view(request: Request, c: Character) -> dict:
    d = c.model_dump()
    jobs = _jobs(request)
    look = jobs.character_lookup(c.series_id)
    d["has_approved_voice"] = look(c.character_id) is not None
    d["episodes"] = _series(request).character_history(c.series_id, c.character_id)
    return d


@router.get("/projects/{project_id}/characters")
def list_characters(project_id: str, request: Request, include_archived: bool = False) -> list[dict]:
    _store(request).get_project(project_id)
    return [_character_view(request, c) for c in _series(request).list_characters(project_id, include_archived)]


@router.post("/projects/{project_id}/characters", status_code=201)
def create_character(project_id: str, body: CharacterCreate, request: Request) -> dict:
    store = _store(request)
    p = store.get_project(project_id)
    if p.kind != "series":
        store.update_project(project_id, kind="series")  # a project with characters is a series
    c = _series(request).create_character(project_id, body.canonical_name, body.aliases, body.description, body.role,
                                          body.gender)
    return _character_view(request, c)


@router.get("/projects/{project_id}/characters/{character_id}")
def get_character(project_id: str, character_id: str, request: Request) -> dict:
    return _character_view(request, _series(request).get_character(project_id, character_id))


@router.patch("/projects/{project_id}/characters/{character_id}")
def update_character(project_id: str, character_id: str, body: CharacterUpdate, request: Request) -> dict:
    c = _series(request).update_character(project_id, character_id, **body.model_dump(exclude_unset=True))
    return _character_view(request, c)


@router.delete("/projects/{project_id}/characters/{character_id}", status_code=204)
def delete_character(project_id: str, character_id: str, request: Request) -> Response:
    series, jobs = _series(request), _jobs(request)
    series.get_character(project_id, character_id)
    open_jobs = []
    for rec in jobs.list(project_id):
        if rec.state in (JobState.AWAITING_REVIEW, JobState.GENERATING) and rec.video_id:
            try:
                d = jobs.load_draft(rec.job_id)
            except NotFoundError:
                continue
            if any(s_.character_id == character_id for s_ in d.speakers):
                open_jobs.append(rec.job_id)
    if open_jobs:
        raise ConflictError(f"Character is assigned in {len(open_jobs)} open job(s); unassign first.")
    series.delete_character(project_id, character_id)
    return Response(status_code=204)


def _check_asset(request: Request, project_id: str, source: VoiceSource, asset_id: Optional[str]) -> TTSEngine:
    """Validate a voice profile's asset and return its engine."""
    if source == VoiceSource.CAST or source == VoiceSource.CLONE:
        if not asset_id:
            raise HTTPException(422, "voice_asset_id (Voice Library voice) is required")
        found = _lib(request).lookup(asset_id)
        if found is None:
            raise HTTPException(422, "voice_asset_id is not an approved, active Voice Library voice")
        return found[1].engine
    if source == VoiceSource.LIBRARY:
        if not asset_id or asset_id not in _store(request).library_ids_with_transcript(project_id):
            raise HTTPException(422, "voice_asset_id must be a project library speaker with a reference recording and transcript")
        return TTSEngine.INDICF5
    if source == VoiceSource.DEFAULT:
        raise HTTPException(422, "Default voice assets use retired XTTS; select a Parler or IndicF5 voice.")
    raise HTTPException(422, "voice_source must be cast, clone, library or default")


@router.post("/projects/{project_id}/characters/{character_id}/voice-profiles", status_code=201)
def add_voice_profile(project_id: str, character_id: str, body: VoiceProfileCreate, request: Request) -> dict:
    engine = _check_asset(request, project_id, body.voice_source, body.voice_asset_id)
    c = _series(request).add_voice_profile(project_id, character_id, body.voice_source, engine, body.voice_asset_id,
                                           body.label, body.notes, body.make_active, body.approved)
    return _character_view(request, c)


@router.patch("/projects/{project_id}/characters/{character_id}/voice-profiles/{voice_profile_id}")
def update_voice_profile(project_id: str, character_id: str, voice_profile_id: str, body: VoiceProfileUpdate,
                         request: Request) -> dict:
    c = _series(request).update_voice_profile(project_id, character_id, voice_profile_id,
                                              **body.model_dump(exclude_unset=True))
    return _character_view(request, c)


# =============================================================================== episodes / identity
class VideoUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    episode_number: Optional[int] = Field(None, ge=0, le=10000)


@router.patch("/projects/{project_id}/videos/{video_id}")
def update_video(project_id: str, video_id: str, body: VideoUpdate, request: Request) -> dict:
    return _store(request).update_video(project_id, video_id, **body.model_dump(exclude_unset=True)).model_dump()


@router.get("/projects/{project_id}/videos/{video_id}/speaker-mapping")
def episode_mapping(project_id: str, video_id: str, request: Request) -> dict:
    _store(request).get_video(project_id, video_id)
    m = _series(request).load_episode_mapping(project_id, video_id)
    if m is None:
        raise HTTPException(404, "No approved speaker mapping for this episode yet")
    return m.model_dump()


@router.get("/jobs/{job_id}/identity")
def job_identity(job_id: str, request: Request) -> dict:
    _job(request, job_id)
    f = _jobs(request).job_output_dir(job_id) / "identity_suggestions.json"
    if not f.is_file():
        raise HTTPException(404, "Identity suggestions are not available for this job")
    return read_json(f)


@router.post("/jobs/{job_id}/identity/refresh", response_model=None)
def refresh_identity(job_id: str, request: Request) -> dict:
    """Re-run suggestions against the current character bank (e.g. after characters were added)."""
    jobs = _jobs(request)
    rec = _job(request, job_id)
    if rec.state != JobState.AWAITING_REVIEW:
        raise ConflictError("Suggestions can only be refreshed while the job is awaiting review.")
    with jobs._lock:
        draft = jobs.load_draft(job_id)
        draft = jobs.suggest_identities(job_id, draft, rec.project_id)
        draft.version += 1
        jobs.projects.save_draft(draft)
    from backend.services import drafts as dr
    return dr.present(draft).model_dump()


# =============================================================================== QA / release / stems
@router.get("/jobs/{job_id}/qa-report")
def qa_report(job_id: str, request: Request) -> dict:
    rec = _job(request, job_id)
    f = Path(rec.qa_report_path) if rec.qa_report_path else _jobs(request).job_output_dir(job_id) / "qa_report.json"
    if not f.is_file():
        raise HTTPException(404, "qa_report.json is not available yet")
    return read_json(f)


@router.get("/jobs/{job_id}/generation-runs")
def generation_runs(job_id: str, request: Request) -> dict:
    _job(request, job_id)
    f = _jobs(request).job_output_dir(job_id) / "generation_runs.json"
    if not f.is_file():
        raise HTTPException(404, "generation_runs.json is not available yet")
    return {"job_id": job_id, "runs": read_json(f)}


class ReleaseRequest(BaseModel):
    approved: bool
    note: str = Field("", max_length=2000)


@router.post("/jobs/{job_id}/release", response_model=JobRecord)
def set_release(job_id: str, body: ReleaseRequest, request: Request) -> JobRecord:
    return _jobs(request).set_release(job_id, body.approved, body.note)


@router.get("/jobs/{job_id}/stems/{stem}")
def job_stem(job_id: str, stem: str, request: Request) -> Response:
    """Listen to original (master), separated vocals/background, Hindi dialogue, processed bed or final mix."""
    rec = _job(request, job_id)
    if stem not in {"master", "vocals", "background", "dialogue", "bed", "mix"}:
        raise HTTPException(404, "Unknown stem")
    f = Path(rec.qa_report_path) if rec.qa_report_path else _jobs(request).job_output_dir(job_id) / "qa_report.json"
    path = None
    if f.is_file():
        path = (read_json(f).get("stems") or {}).get(stem)
    if path is None and stem in {"master", "vocals", "background"}:
        path = _jobs(request).analysis_index(job_id).get(stem)
    if not path:
        raise HTTPException(404, f"Stem '{stem}' is not available for this job")
    return _stream_file(request, Path(path), "audio/wav")
