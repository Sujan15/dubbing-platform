# hindi-dubbing-platform/backend/services/projects.py
"""Project, video and Speaker Library persistence (JSON documents + WAV files, no database server).

Layout under ``DATA_DIR/projects``::

    {project_id}/project.json
    {project_id}/speakers/{speaker_id}/metadata.json | reference.wav | history/
    {project_id}/videos/{video_id}/video.json | original/input.<ext> | drafts/draft.json | drafts/approved_v<n>.json

Ids are uuid4 hex strings validated before they touch a path. Display names are never identifiers.
Uploaded originals are immutable; a replaced library reference is moved to ``history/`` instead of
being overwritten.
"""

from __future__ import annotations

import logging
import shutil
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.app.config import TARGET_LANGUAGE, MixMode, Settings, TranslationEngine
from backend.services.drafts import DraftManifest
from backend.services.media import FFmpeg
from backend.services.storage import (
    ConflictError,
    NotFoundError,
    ValidationFailed,
    atomic_write_json,
    new_id,
    now,
    read_json,
    require_entity_id,
    write_once_json,
)
from backend.services.voices import ReferenceReport, normalize_reference

log = logging.getLogger("dubbing.projects")


class ProjectSettings(BaseModel):
    source_language: Optional[str] = None
    mix_mode: MixMode = MixMode.PRESERVE_BACKGROUND
    translation_engine: Optional[TranslationEngine] = None
    review_enabled: Optional[bool] = None  # None = PROJECT_REVIEW_ENABLED


class Project(BaseModel):
    project_id: str
    name: str
    description: str = ""
    target_language: str = TARGET_LANGUAGE
    settings: ProjectSettings = Field(default_factory=ProjectSettings)
    status: str = "active"
    kind: Literal["standalone", "series"] = "standalone"   # v3: a series shares characters across episodes
    created_at: datetime
    updated_at: datetime


class ProjectVideo(BaseModel):
    video_id: str
    project_id: str
    title: str
    original_filename: str
    stored_filename: str
    size_bytes: int
    duration: float
    width: int = 0
    height: int = 0
    fps: float = 0.0
    job_ids: list[str] = Field(default_factory=list)
    current_job_id: Optional[str] = None
    episode_number: Optional[int] = None                      # v3: series episode
    created_at: datetime
    updated_at: datetime


class SpeakerSource(str, Enum):
    CLONED = "cloned"
    UPLOADED = "uploaded"
    DEFAULT_PROMOTED = "default_promoted"


class ProjectSpeaker(BaseModel):
    speaker_id: str
    project_id: str
    display_name: str
    source: Optional[SpeakerSource] = None
    reference_available: bool = False
    reference_duration_seconds: float = 0.0
    reference_speech_seconds: float = 0.0
    reference_sample_rate: Optional[int] = None
    reference_transcript: Optional[str] = None
    reference_updated_at: Optional[datetime] = None
    notes: str = ""
    origin: dict = Field(default_factory=dict)  # e.g. {"job_id", "video_id", "job_speaker_id", "default_voice_id"}
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ProjectStore:
    def __init__(self, settings: Settings, ff: FFmpeg):
        self.s = settings
        self.ff = ff
        self.root = settings.projects_dir
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ paths
    def project_dir(self, project_id: str) -> Path:
        return self.root / require_entity_id(project_id, "Project")

    def video_dir(self, project_id: str, video_id: str) -> Path:
        return self.project_dir(project_id) / "videos" / require_entity_id(video_id, "Video")

    def speaker_dir(self, project_id: str, speaker_id: str) -> Path:
        return self.project_dir(project_id) / "speakers" / require_entity_id(speaker_id, "Speaker")

    def drafts_dir(self, project_id: str, video_id: str) -> Path:
        return self.video_dir(project_id, video_id) / "drafts"

    # ------------------------------------------------------------------ projects
    def create_project(self, name: str, description: str = "", target_language: str = TARGET_LANGUAGE,
                       settings: Optional[ProjectSettings] = None, kind: str = "standalone") -> Project:
        name = name.strip()
        with self._lock:
            if any(p.name.casefold() == name.casefold() for p in self.list_projects()):
                raise ConflictError(f"A project named '{name}' already exists.")
            ts = now()
            p = Project(project_id=new_id(), name=name, description=description.strip(),
                        target_language=target_language, settings=settings or ProjectSettings(), kind=kind,
                        created_at=ts, updated_at=ts)
            atomic_write_json(self.project_dir(p.project_id) / "project.json", p)
        log.info("Project %s created (%s)", p.project_id, p.name)
        return p

    def list_projects(self) -> list[Project]:
        out = []
        if not self.root.is_dir():
            return out
        for f in self.root.glob("*/project.json"):
            if f.parent.name.startswith("."):
                continue
            try:
                out.append(Project.model_validate(read_json(f)))
            except (ValueError, OSError) as exc:
                log.warning("Ignoring unreadable project %s: %s", f.parent.name, exc)
        return sorted(out, key=lambda p: p.updated_at, reverse=True)

    def get_project(self, project_id: str) -> Project:
        f = self.project_dir(project_id) / "project.json"
        if not f.is_file():
            raise NotFoundError("Project not found")
        return Project.model_validate(read_json(f))

    def update_project(self, project_id: str, **changes) -> Project:
        with self._lock:
            p = self.get_project(project_id)
            if "name" in changes and changes["name"] is not None:
                name = changes["name"].strip()
                if any(o.project_id != project_id and o.name.casefold() == name.casefold() for o in self.list_projects()):
                    raise ConflictError(f"A project named '{name}' already exists.")
                changes["name"] = name
            data = p.model_dump() | {k: v for k, v in changes.items() if v is not None} | {"updated_at": now()}
            p = Project.model_validate(data)
            atomic_write_json(self.project_dir(project_id) / "project.json", p)
            return p

    def touch_project(self, project_id: str) -> None:
        try:
            self.update_project(project_id)
        except NotFoundError:
            pass

    def delete_project(self, project_id: str) -> None:
        d = self.project_dir(project_id)
        if not (d / "project.json").is_file():
            raise NotFoundError("Project not found")
        with self._lock:
            trash = d.with_name(f".deleting-{project_id}")
            try:
                d.rename(trash)  # atomic: the project disappears at once, files are removed afterwards
            except OSError as exc:  # Windows: a file is open (e.g. a video being streamed)
                raise ConflictError(f"Project files are in use; close previews and retry ({exc.strerror}).") from exc
            shutil.rmtree(trash, ignore_errors=True)
        log.info("Project %s deleted", project_id)

    # ------------------------------------------------------------------ videos
    def new_video_slot(self, project_id: str, ext: str) -> tuple[str, Path]:
        self.get_project(project_id)
        video_id = new_id()
        original = self.video_dir(project_id, video_id) / "original"
        original.mkdir(parents=True, exist_ok=False)
        return video_id, original / f"input{ext}"

    def add_video(self, video: ProjectVideo) -> ProjectVideo:
        atomic_write_json(self.video_dir(video.project_id, video.video_id) / "video.json", video)
        self.touch_project(video.project_id)
        return video

    def discard_video_slot(self, project_id: str, video_id: str) -> None:
        shutil.rmtree(self.video_dir(project_id, video_id), ignore_errors=True)

    def list_videos(self, project_id: str) -> list[ProjectVideo]:
        base = self.project_dir(project_id) / "videos"
        out = []
        for f in base.glob("*/video.json") if base.is_dir() else []:
            try:
                out.append(ProjectVideo.model_validate(read_json(f)))
            except (ValueError, OSError) as exc:
                log.warning("Ignoring unreadable video record %s: %s", f, exc)
        return sorted(out, key=lambda v: v.created_at)

    def get_video(self, project_id: str, video_id: str) -> ProjectVideo:
        f = self.video_dir(project_id, video_id) / "video.json"
        if not f.is_file():
            raise NotFoundError("Video not found")
        return ProjectVideo.model_validate(read_json(f))

    def update_video(self, project_id: str, video_id: str, **changes) -> ProjectVideo:
        with self._lock:
            v = self.get_video(project_id, video_id)
            if changes.get("episode_number") is not None and any(
                    o.video_id != video_id and o.episode_number == changes["episode_number"]
                    for o in self.list_videos(project_id)):
                raise ConflictError(f"Episode {changes['episode_number']} already exists in this project.")
            data = v.model_dump() | {k: val for k, val in changes.items() if val is not None} | {"updated_at": now()}
            return self.add_video(ProjectVideo.model_validate(data))

    def attach_job(self, project_id: str, video_id: str, job_id: str) -> ProjectVideo:
        with self._lock:
            v = self.get_video(project_id, video_id)
            v = v.model_copy(update={"job_ids": [*v.job_ids, job_id], "current_job_id": job_id, "updated_at": now()})
            return self.add_video(v)

    # ------------------------------------------------------------------ drafts
    def load_draft(self, project_id: str, video_id: str) -> DraftManifest:
        f = self.drafts_dir(project_id, video_id) / "draft.json"
        if not f.is_file():
            raise NotFoundError("Draft not found")
        return DraftManifest.model_validate(read_json(f))

    def save_draft(self, draft: DraftManifest) -> DraftManifest:
        atomic_write_json(self.drafts_dir(draft.project_id, draft.video_id) / "draft.json", draft)
        return draft

    def write_approved_snapshot(self, draft: DraftManifest) -> Path:
        path = self.drafts_dir(draft.project_id, draft.video_id) / f"approved_v{draft.version}.json"
        return write_once_json(path, draft)

    def load_approved(self, project_id: str, video_id: str, version: int) -> DraftManifest:
        f = self.drafts_dir(project_id, video_id) / f"approved_v{int(version)}.json"
        if not f.is_file():
            raise NotFoundError("Approved draft snapshot not found")
        return DraftManifest.model_validate(read_json(f))

    # ------------------------------------------------------------------ speaker library
    def list_speakers(self, project_id: str) -> list[ProjectSpeaker]:
        base = self.project_dir(project_id) / "speakers"
        out = []
        for f in base.glob("*/metadata.json") if base.is_dir() else []:
            try:
                out.append(ProjectSpeaker.model_validate(read_json(f)))
            except (ValueError, OSError) as exc:
                log.warning("Ignoring unreadable speaker %s: %s", f, exc)
        return sorted(out, key=lambda s: s.display_name.casefold())

    def get_speaker(self, project_id: str, speaker_id: str) -> ProjectSpeaker:
        self.get_project(project_id)
        f = self.speaker_dir(project_id, speaker_id) / "metadata.json"
        if not f.is_file():
            raise NotFoundError("Speaker not found")
        return ProjectSpeaker.model_validate(read_json(f))

    def _check_name(self, project_id: str, name: str, exclude: Optional[str] = None) -> str:
        name = name.strip()
        if not name:
            raise ValidationFailed("Speaker name is required")
        if any(s.speaker_id != exclude and s.display_name.casefold() == name.casefold()
               for s in self.list_speakers(project_id)):
            raise ConflictError(f"A speaker named '{name}' already exists in this project.")
        return name

    def create_speaker(self, project_id: str, display_name: str, notes: str = "") -> ProjectSpeaker:
        self.get_project(project_id)
        with self._lock:
            name = self._check_name(project_id, display_name)
            ts = now()
            spk = ProjectSpeaker(speaker_id=new_id(), project_id=project_id, display_name=name, notes=notes.strip(),
                                 created_at=ts, updated_at=ts)
            atomic_write_json(self.speaker_dir(project_id, spk.speaker_id) / "metadata.json", spk)
        self.touch_project(project_id)
        return spk

    def update_speaker(self, project_id: str, speaker_id: str, display_name: Optional[str] = None,
                       notes: Optional[str] = None, reference_transcript: Optional[str] = None) -> ProjectSpeaker:
        with self._lock:
            spk = self.get_speaker(project_id, speaker_id)
            changes: dict = {"updated_at": now()}
            if display_name is not None:
                changes["display_name"] = self._check_name(project_id, display_name, exclude=speaker_id)
            if notes is not None:
                changes["notes"] = notes.strip()
            if reference_transcript is not None:
                changes["reference_transcript"] = reference_transcript.strip() or None
            spk = spk.model_copy(update=changes)
            atomic_write_json(self.speaker_dir(project_id, speaker_id) / "metadata.json", spk)
            return spk

    def delete_speaker(self, project_id: str, speaker_id: str) -> None:
        self.get_speaker(project_id, speaker_id)
        with self._lock:
            shutil.rmtree(self.speaker_dir(project_id, speaker_id), ignore_errors=True)
        self.touch_project(project_id)

    def reference_path(self, project_id: str, speaker_id: str) -> Path:
        return self.speaker_dir(project_id, speaker_id) / "reference.wav"

    def set_reference(self, project_id: str, speaker_id: str, src: Path, source: SpeakerSource,
                      replace: bool, origin: Optional[dict] = None,
                      transcript: Optional[str] = None) -> tuple[ProjectSpeaker, ReferenceReport]:
        """Validate + normalise ``src`` and install it as the speaker's reference.

        An existing reference is only replaced when ``replace`` is true; the previous file is kept in
        ``history/`` so a mistaken replacement can be recovered by hand.
        """
        with self._lock:
            spk = self.get_speaker(project_id, speaker_id)
            d = self.speaker_dir(project_id, speaker_id)
            final = d / "reference.wav"
            if final.is_file() and not replace:
                raise ConflictError("This speaker already has a reference; confirm replacement to overwrite it.")
            staged = d / "reference.incoming.wav"
            try:
                report = normalize_reference(self.ff, src, staged, self.s)
                if final.is_file():
                    hist = d / "history"
                    hist.mkdir(exist_ok=True)
                    shutil.move(str(final), str(hist / f"reference_{now().strftime('%Y%m%dT%H%M%S%f')}.wav"))
                staged.replace(final)
            finally:
                staged.unlink(missing_ok=True)
            ts = now()
            spk = spk.model_copy(update={
                "source": source, "reference_available": True, "reference_duration_seconds": report.duration_seconds,
                "reference_speech_seconds": report.speech_seconds, "reference_sample_rate": report.sample_rate,
                "reference_transcript": transcript.strip() if transcript and transcript.strip() else None,
                "reference_updated_at": ts, "updated_at": ts, "warnings": report.warnings,
                "origin": origin or {}})
            atomic_write_json(d / "metadata.json", spk)
        self.touch_project(project_id)
        return spk, report

    def library_lookup(self, project_id: str):
        """Resolver used at generation time: only speakers of *this* project, only validated references."""
        def lookup(speaker_id: str):
            try:
                spk = self.get_speaker(project_id, speaker_id)
            except NotFoundError:
                return None
            ref = self.reference_path(project_id, speaker_id)
            if not spk.reference_available or not ref.is_file():
                return None
            return ref, spk.reference_speech_seconds, spk.display_name, spk.reference_transcript
        return lookup

    def library_ids_with_reference(self, project_id: str) -> set[str]:
        return {s.speaker_id for s in self.list_speakers(project_id)
                if s.reference_available and self.reference_path(project_id, s.speaker_id).is_file()}

    def library_ids_with_transcript(self, project_id: str) -> set[str]:
        return {s.speaker_id for s in self.list_speakers(project_id)
                if s.reference_available and s.reference_transcript and
                self.reference_path(project_id, s.speaker_id).is_file()}
