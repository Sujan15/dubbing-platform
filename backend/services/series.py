# hindi-dubbing-platform/backend/services/series.py
"""Series & character identity: canonical characters with permanent voice profiles, per-episode
speaker mappings, and the character embedding bank used for cross-episode suggestions.

One project == one series (``Project.kind == "series"``); episodes are the project's videos with an
optional ``episode_number``. Existing standalone projects are untouched: a project without
``characters/`` simply has no characters.

Layout under ``projects/{project_id}``::

    characters/{character_id}/character.json     canonical character + its voice profiles
    characters/{character_id}/embeddings.json    {"vectors": [[...], ...], "sources": [...]} (<= MAX_VECTORS)
    characters/{character_id}/generation_log.jsonl
    videos/{video_id}/speaker_mapping.json       the mapping approved for that episode (written on confirm)

INVARIANT: a diarization label (SPEAKER_02) is an *episode-local* id. Only ``character_id`` is a
permanent identity, and only a user-confirmed mapping ever links the two.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from backend.app.config import Settings, TTSEngine
from backend.services.storage import (
    ConflictError,
    NotFoundError,
    ValidationFailed,
    atomic_write_json,
    new_id,
    now,
    read_json,
    require_entity_id,
)
from backend.services.voices import VoiceSource

log = logging.getLogger("dubbing.series")

MAX_VECTORS = 8


class MappingStatus(str, Enum):
    UNASSIGNED = "unassigned"     # no evidence / too little speech: human must decide
    SUGGESTED = "suggested"       # embedding match in [review, high) or high but unconfirmed
    CONFIRMED = "confirmed"       # user confirmed in Human Review
    REJECTED = "rejected"         # user rejected the suggestion


class VoiceProfile(BaseModel):
    voice_profile_id: str
    character_id: str
    voice_source: VoiceSource                 # cast | library(project speaker) | default | clone(voice library clone)
    engine: TTSEngine
    voice_asset_id: Optional[str] = None      # voice library voice_id / project speaker_id / default voice id
    label: str = ""
    identity_locked: bool = False             # locked profiles are never replaced without an explicit unlock
    approved: bool = False
    notes: str = ""
    created_at: datetime
    updated_at: datetime


class Character(BaseModel):
    character_id: str
    series_id: str                            # == project_id
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    role: Optional[str] = None                # hero | heroine | villain | support | narrator | other
    gender: Optional[str] = None
    status: str = "active"
    voice_profiles: list[VoiceProfile] = Field(default_factory=list)
    active_voice_profile_id: Optional[str] = None
    embedding_count: int = 0
    created_at: datetime
    updated_at: datetime

    def active_profile(self) -> Optional[VoiceProfile]:
        return next((p for p in self.voice_profiles if p.voice_profile_id == self.active_voice_profile_id), None)


class SpeakerMapping(BaseModel):
    local_speaker_id: str
    character_id: Optional[str] = None
    confidence: Optional[float] = None
    mapping_status: MappingStatus = MappingStatus.UNASSIGNED
    reviewed_by_user: bool = False
    evidence: dict = Field(default_factory=dict)   # {"speech_seconds", "candidates": [...], "reason"}


class EpisodeSpeakerMapping(BaseModel):
    project_id: str
    video_id: str
    job_id: str
    episode_number: Optional[int] = None
    approved_version: Optional[int] = None
    mappings: list[SpeakerMapping]
    created_at: datetime


class SeriesStore:
    def __init__(self, settings: Settings, projects_root: Path):
        self.s = settings
        self.root = projects_root
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ paths
    def _project_dir(self, project_id: str) -> Path:
        d = self.root / require_entity_id(project_id, "Project")
        if not (d / "project.json").is_file():
            raise NotFoundError("Project not found")
        return d

    def character_dir(self, project_id: str, character_id: str) -> Path:
        return self._project_dir(project_id) / "characters" / require_entity_id(character_id, "Character")

    def _save(self, c: Character) -> Character:
        atomic_write_json(self.character_dir(c.series_id, c.character_id) / "character.json", c)
        return c

    # ------------------------------------------------------------------ characters
    def list_characters(self, project_id: str, include_archived: bool = False) -> list[Character]:
        base = self._project_dir(project_id) / "characters"
        out = []
        for f in base.glob("*/character.json") if base.is_dir() else []:
            try:
                c = Character.model_validate(read_json(f))
            except (ValueError, OSError) as exc:
                log.warning("Ignoring unreadable character %s: %s", f, exc)
                continue
            if c.status == "active" or include_archived:
                out.append(c)
        return sorted(out, key=lambda c: c.canonical_name.casefold())

    def get_character(self, project_id: str, character_id: str) -> Character:
        f = self.character_dir(project_id, character_id) / "character.json"
        if not f.is_file():
            raise NotFoundError("Character not found")
        return Character.model_validate(read_json(f))

    def _check_name(self, project_id: str, name: str, aliases: list[str], exclude: Optional[str] = None) -> str:
        name = name.strip()
        if not name:
            raise ValidationFailed("canonical_name is required")
        taken = {n.casefold() for c in self.list_characters(project_id, include_archived=True) if c.character_id != exclude
                 for n in [c.canonical_name, *c.aliases]}
        for n in [name, *aliases]:
            if n.casefold() in taken:
                raise ConflictError(f"'{n}' is already a character name or alias in this series.")
        return name

    def create_character(self, project_id: str, canonical_name: str, aliases: Optional[list[str]] = None,
                         description: str = "", role: Optional[str] = None, gender: Optional[str] = None) -> Character:
        aliases = [a.strip() for a in (aliases or []) if a.strip()]
        with self._lock:
            name = self._check_name(project_id, canonical_name, aliases)
            ts = now()
            c = Character(character_id=new_id(), series_id=project_id, canonical_name=name, aliases=aliases,
                          description=description.strip(), role=role, gender=gender, created_at=ts, updated_at=ts)
            self.character_dir(project_id, c.character_id).mkdir(parents=True, exist_ok=True)
            return self._save(c)

    def update_character(self, project_id: str, character_id: str, **changes) -> Character:
        with self._lock:
            c = self.get_character(project_id, character_id)
            if changes.get("aliases") is not None:
                changes["aliases"] = [a.strip() for a in changes["aliases"] if a.strip()]
            if changes.get("canonical_name") is not None or changes.get("aliases") is not None:
                changes["canonical_name"] = self._check_name(
                    project_id, changes.get("canonical_name") or c.canonical_name,
                    changes.get("aliases") if changes.get("aliases") is not None else c.aliases, exclude=character_id)
            data = c.model_dump() | {k: v for k, v in changes.items() if v is not None} | {"updated_at": now()}
            return self._save(Character.model_validate(data))

    def delete_character(self, project_id: str, character_id: str) -> None:
        self.get_character(project_id, character_id)
        with self._lock:
            shutil.rmtree(self.character_dir(project_id, character_id), ignore_errors=True)

    # ------------------------------------------------------------------ voice profiles
    def add_voice_profile(self, project_id: str, character_id: str, voice_source: VoiceSource, engine: TTSEngine,
                          voice_asset_id: Optional[str], label: str = "", notes: str = "",
                          make_active: bool = True, approved: bool = False) -> Character:
        with self._lock:
            c = self.get_character(project_id, character_id)
            active = c.active_profile()
            if make_active and active and active.identity_locked:
                raise ConflictError(f"{c.canonical_name}: the active voice profile is identity-locked; unlock it "
                                    "explicitly before assigning another voice.")
            ts = now()
            p = VoiceProfile(voice_profile_id=new_id(), character_id=character_id, voice_source=voice_source,
                             engine=engine, voice_asset_id=voice_asset_id, label=label.strip(), notes=notes.strip(),
                             approved=approved, created_at=ts, updated_at=ts)
            c.voice_profiles.append(p)
            if make_active or not c.active_voice_profile_id:
                c.active_voice_profile_id = p.voice_profile_id
            c.updated_at = ts
            return self._save(c)

    def update_voice_profile(self, project_id: str, character_id: str, voice_profile_id: str, **changes) -> Character:
        with self._lock:
            c = self.get_character(project_id, character_id)
            p = next((p for p in c.voice_profiles if p.voice_profile_id == voice_profile_id), None)
            if p is None:
                raise NotFoundError("Voice profile not found")
            make_active = changes.pop("make_active", None)
            for k, v in changes.items():
                if v is not None:
                    setattr(p, k, v)
            p.updated_at = now()
            if make_active:
                active = c.active_profile()
                if active and active is not p and active.identity_locked and not changes.get("identity_locked") is False:
                    raise ConflictError(f"{c.canonical_name}: the active voice profile is identity-locked.")
                c.active_voice_profile_id = p.voice_profile_id
            c.updated_at = now()
            return self._save(c)

    def profiles_using(self, project_id: str, voice_source: VoiceSource, asset_id: str) -> list[str]:
        return [f"{c.canonical_name}/{p.voice_profile_id[:8]}" for c in self.list_characters(project_id, True)
                for p in c.voice_profiles if p.voice_source == voice_source and p.voice_asset_id == asset_id]

    # ------------------------------------------------------------------ embeddings
    def _emb_path(self, project_id: str, character_id: str) -> Path:
        return self.character_dir(project_id, character_id) / "embeddings.json"

    def character_embeddings(self, project_id: str) -> dict[str, np.ndarray]:
        """character_id -> (n, d) unit vectors; characters without evidence are absent."""
        out = {}
        for c in self.list_characters(project_id):
            f = self._emb_path(project_id, c.character_id)
            if not f.is_file():
                continue
            try:
                vecs = np.asarray(read_json(f).get("vectors") or [], dtype=np.float32)
            except (ValueError, OSError):
                continue
            if vecs.ndim == 2 and len(vecs):
                out[c.character_id] = vecs
        return out

    def add_character_embedding(self, project_id: str, character_id: str, vector: np.ndarray, source: dict) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            return
        vector = vector / norm
        with self._lock:
            f = self._emb_path(project_id, character_id)
            data = read_json(f) if f.is_file() else {"vectors": [], "sources": []}
            if any(s.get("job_id") == source.get("job_id") and s.get("local_speaker_id") == source.get("local_speaker_id")
                   for s in data["sources"]):
                return  # same evidence already recorded (retry / re-confirm)
            data["vectors"] = (data["vectors"] + [vector.round(6).tolist()])[-MAX_VECTORS:]
            data["sources"] = (data["sources"] + [source])[-MAX_VECTORS:]
            atomic_write_json(f, data)
            c = self.get_character(project_id, character_id)
            self._save(c.model_copy(update={"embedding_count": len(data["vectors"]), "updated_at": now()}))

    # ------------------------------------------------------------------ episode mappings
    def mapping_path(self, project_id: str, video_id: str) -> Path:
        return self._project_dir(project_id) / "videos" / require_entity_id(video_id, "Video") / "speaker_mapping.json"

    def write_episode_mapping(self, m: EpisodeSpeakerMapping) -> EpisodeSpeakerMapping:
        atomic_write_json(self.mapping_path(m.project_id, m.video_id), m)
        return m

    def load_episode_mapping(self, project_id: str, video_id: str) -> Optional[EpisodeSpeakerMapping]:
        f = self.mapping_path(project_id, video_id)
        if not f.is_file():
            return None
        return EpisodeSpeakerMapping.model_validate(read_json(f))

    def character_history(self, project_id: str, character_id: str) -> list[dict]:
        """Episodes in which this character was confirmed, for the review panel."""
        base = self._project_dir(project_id) / "videos"
        hits = []
        for f in base.glob("*/speaker_mapping.json") if base.is_dir() else []:
            try:
                m = EpisodeSpeakerMapping.model_validate(read_json(f))
            except (ValueError, OSError):
                continue
            for sm in m.mappings:
                if sm.character_id == character_id and sm.mapping_status == MappingStatus.CONFIRMED:
                    hits.append({"video_id": m.video_id, "job_id": m.job_id, "episode_number": m.episode_number,
                                 "local_speaker_id": sm.local_speaker_id, "confirmed_at": m.created_at})
        return sorted(hits, key=lambda h: (h["episode_number"] or 10**6, h["confirmed_at"]))

    def log_generation(self, project_id: str, character_id: str, entry: dict) -> None:
        with open(self.character_dir(project_id, character_id) / "generation_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
