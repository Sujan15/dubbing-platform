# hindi-dubbing-platform/backend/services/voice_library.py

"""Voice Library for IndicF5 clone voices and Parler cast presets.

Layout under ``VOICE_LIBRARY_DIR``::

    {voice_id}/voice.json           metadata (engine, prompt, model pin, licence, approval, preview)
    {voice_id}/reference.wav        normalised master reference (reference-conditioned engines)
    {voice_id}/preview.wav          last audition
    {voice_id}/history/             replaced references (never deleted automatically)
    {voice_id}/runs.jsonl           every audition / generation run that used this voice

Rules enforced here, not in the UI:

* A voice is selectable for generation only when ``approved`` and ``active``.
* Approval is bound to a configuration hash (prompt, reference content, model id/revision, params,
  seed). Changing any of them clears ``approved``; approval requires an audition generated from the
  *current* configuration, so nobody approves a voice they have not heard.
* An approved master reference is never replaced without ``replace=True``; the old file is kept.
* The pipeline resolves a voice to a ``VoiceSpec`` through ``lookup`` - never by reading files itself.

Historical XTTS records remain readable, but cannot be auditioned, approved, or selected.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from backend.app.config import TARGET_LANGUAGE, Settings, TTSEngine
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
)
from backend.services.tts import VoiceSpec, engine_class
from backend.services.voices import ReferenceReport, normalize_reference

log = logging.getLogger("dubbing.voice_library")

VoiceKind = Literal["clone", "cast", "legacy"]
CATALOG_SLOT_RE = r"^[A-Z][A-Z0-9_]{2,63}$"

# Curated Parler catalog slots are created inactive and unapproved until auditioned and approved.
CATALOG_SLOTS: list[dict] = [
    {
        "slot": "HIN_MALE_HERO_01",
        "voice_name": "Hero — Rohit",
        "gender": "male",
        "description": "Hero",
        "prompt": "Rohit speaks with a balanced, confident male pitch and a natural, expressive tone at a moderate pace. The recording is of very high quality, sounding very clear and close up with almost no background noise.",
    },
    {
        "slot": "HIN_FEMALE_HEROINE_01",
        "voice_name": "Heroine — Divya",
        "gender": "female",
        "description": "Heroine",
        "prompt": "Divya speaks with a warm, slightly high-pitched female voice and an expressive, engaging tone at a moderate pace. The recording is very clear audio, close-sounding, and captured in a studio environment with no background noise.",
    },
    {
        "slot": "HIN_MALE_VILLAIN_01",
        "voice_name": "Villain — Aman",
        "gender": "male",
        "description": "Villain",
        "prompt": "Aman's deep, low-pitched male voice delivers speech with an intense, slightly slower, and authoritative tone. The recording is very clear audio, close-sounding, with no background noise.",
    },
    {
        "slot": "HIN_FEMALE_SUPPORT_01",
        "voice_name": "Supporting Female — Rani",
        "gender": "female",
        "description": "Supporting Female",
        "prompt": "Rani speaks with a natural female pitch and a calm, conversational, slightly expressive tone at a normal pace. The recording is of very high quality, very clear audio, and close-sounding with no background noise.",
    },
    {
        "slot": "HIN_MALE_SUPPORT_01",
        "voice_name": "Supporting Male — Rohit",
        "gender": "male",
        "description": "Supporting Male",
        "prompt": "Rohit speaks at a moderate pace with a neutral, conversational male pitch and a slightly monotone, steady delivery. The recording is very clear audio, close-sounding, with no background noise.",
    },
]
DEFAULT_AUDITION_TEXT = "नमस्ते, मेरा नाम अर्जुन है। आज हम एक नई कहानी की शुरुआत करने जा रहे हैं।"


class VoicePreview(BaseModel):
    text: str
    generated_at: datetime
    config_hash: str
    prompt_version: int = 0
    artifact_name: Optional[str] = None
    engine: TTSEngine
    model_id: str
    model_revision: Optional[str] = None
    duration_seconds: float
    run_id: str
    problems: list[str] = Field(default_factory=list)


class VoicePromptVersion(BaseModel):
    version: int
    prompt: str
    created_at: datetime


class VoiceRecord(BaseModel):
    voice_id: str
    voice_name: str
    kind: VoiceKind
    engine: TTSEngine
    language: str = TARGET_LANGUAGE
    gender: Optional[str] = None
    description: str = ""
    catalog_slot: Optional[str] = None
    # reference-conditioned engine (IndicF5 clone)
    reference_available: bool = False
    reference_transcript: Optional[str] = None
    reference_language: str = TARGET_LANGUAGE
    reference_duration_seconds: float = 0.0
    reference_speech_seconds: float = 0.0
    reference_sample_rate: Optional[int] = None
    reference_quality: list[str] = Field(default_factory=list)
    reference_sha256: Optional[str] = None
    # prompt-conditioned engines (parler cast)
    voice_description_prompt: Optional[str] = None
    prompt_version: int = 0
    prompt_versions: list[VoicePromptVersion] = Field(default_factory=list)
    sample_text: str = DEFAULT_AUDITION_TEXT
    generation_parameters: dict = Field(default_factory=dict)
    seed: Optional[int] = None
    # model pin + licence
    model_id: str = ""
    model_revision: Optional[str] = None
    supported_language_status: Literal["unvalidated", "validated", "rejected"] = "unvalidated"
    license_metadata: dict = Field(default_factory=dict)
    quality_benchmark: dict = Field(default_factory=dict)
    # approval
    approved: bool = False
    approved_at: Optional[datetime] = None
    approved_config_hash: Optional[str] = None
    approval_note: str = ""
    active: bool = True
    preview: Optional[VoicePreview] = None
    origin: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def preserve_legacy_prompt_version(cls, values):
        if not isinstance(values, dict):
            return values
        prompt = (values.get("voice_description_prompt") or "").strip()
        versions = values.get("prompt_versions") or []
        if prompt and not versions:
            values = dict(values)
            version = values.get("prompt_version") or 1
            values["prompt_version"] = version
            values["prompt_versions"] = [{
                "version": version,
                "prompt": prompt,
                "created_at": values.get("updated_at") or values.get("created_at") or now(),
            }]
        elif versions and not values.get("prompt_version"):
            values = dict(values)
            values["prompt_version"] = max(revision.get("version", 0) for revision in versions)
        return values

    # ---- derived ------------------------------------------------------------------
    def config_hash(self) -> str:
        src = {"engine": self.engine.value, "prompt": self.voice_description_prompt, "ref": self.reference_sha256,
               "transcript": self.reference_transcript, "model": self.model_id, "revision": self.model_revision,
               "params": self.generation_parameters, "seed": self.seed}
        return hashlib.sha256(json.dumps(src, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]

    @property
    def selectable(self) -> bool:
        return (self.engine != TTSEngine.XTTS and self.approved and self.active and
                self.approved_config_hash == self.config_hash())

    def public(self) -> dict:
        d = self.model_dump()
        d["config_hash"] = self.config_hash()
        d["selectable"] = self.selectable
        d["needs_reaudition"] = self.approved and self.approved_config_hash != self.config_hash()
        return d


class VoiceLibrary:
    def __init__(self, settings: Settings, ff: FFmpeg):
        self.s = settings
        self.ff = ff
        self.root = Path(settings.voice_library_dir)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ paths / io
    def voice_dir(self, voice_id: str) -> Path:
        return self.root / require_entity_id(voice_id, "Voice")

    def reference_path(self, voice_id: str) -> Path:
        return self.voice_dir(voice_id) / "reference.wav"

    def preview_path(self, voice_id: str) -> Path:
        return self.voice_dir(voice_id) / "preview.wav"

    def audition_path(self, voice_id: str, run_id: str) -> Path:
        return self.voice_dir(voice_id) / "auditions" / f"{require_entity_id(run_id, 'Audition')}.wav"

    def _save(self, v: VoiceRecord) -> VoiceRecord:
        atomic_write_json(self.voice_dir(v.voice_id) / "voice.json", v)
        return v

    def get(self, voice_id: str) -> VoiceRecord:
        f = self.voice_dir(voice_id) / "voice.json"
        if not f.is_file():
            raise NotFoundError("Voice not found")
        return VoiceRecord.model_validate(read_json(f))

    def list(self, engine: Optional[TTSEngine] = None, kind: Optional[str] = None, gender: Optional[str] = None,
             approved: Optional[bool] = None, include_inactive: bool = True) -> list[VoiceRecord]:
        out = []
        for f in self.root.glob("*/voice.json") if self.root.is_dir() else []:
            try:
                v = VoiceRecord.model_validate(read_json(f))
            except (ValueError, OSError) as exc:
                log.warning("Ignoring unreadable voice %s: %s", f.parent.name, exc)
                continue
            if engine and v.engine != engine or kind and v.kind != kind:
                continue
            if gender and (v.gender or "").casefold() != gender.casefold():
                continue
            if approved is not None and v.approved != approved:
                continue
            if not include_inactive and not v.active:
                continue
            out.append(v)
        return sorted(out, key=lambda v: (v.catalog_slot or "~", v.voice_name.casefold()))

    # ------------------------------------------------------------------ create / edit
    def _check_name(self, name: str, exclude: Optional[str] = None) -> str:
        name = name.strip()
        if not name:
            raise ValidationFailed("voice_name is required")
        if any(v.voice_id != exclude and v.voice_name.casefold() == name.casefold() for v in self.list()):
            raise ConflictError(f"A voice named '{name}' already exists in the library.")
        return name

    def _engine_defaults(self, engine: TTSEngine) -> tuple[str, Optional[str], dict]:
        if engine == TTSEngine.XTTS:
            raise ValidationFailed("XTTS has been removed; create a Parler or IndicF5 voice.")
        caps = engine_class(engine).get_capabilities()
        model_id, revision = {
            TTSEngine.INDICF5: (self.s.indicf5_model, self.s.indicf5_revision),
            TTSEngine.PARLER: (self.s.parler_model, self.s.parler_revision),
        }[engine]
        return model_id, revision, {"license": caps.license, "commercial_use": caps.commercial_use}

    def create(self, *, voice_name: str, engine: TTSEngine, gender: Optional[str] = None, description: str = "",
               voice_description_prompt: Optional[str] = None, sample_text: Optional[str] = None,
               generation_parameters: Optional[dict] = None, seed: Optional[int] = None,
               catalog_slot: Optional[str] = None, reference_transcript: Optional[str] = None,
               origin: Optional[dict] = None) -> VoiceRecord:
        engine = TTSEngine(engine)
        caps = engine_class(engine).get_capabilities()
        if caps.prompt_conditioned and not (voice_description_prompt or "").strip():
            raise ValidationFailed(f"{engine.value} voices need a voice_description_prompt")
        if caps.supports_seed is False and seed is not None:
            raise ValidationFailed(f"{engine.value} does not support a fixed seed")
        kind: VoiceKind = "cast" if caps.prompt_conditioned else "clone"
        with self._lock:
            name = self._check_name(voice_name)
            if catalog_slot and any(v.catalog_slot == catalog_slot for v in self.list()):
                raise ConflictError(f"Catalog slot {catalog_slot} is already taken.")
            model_id, revision, licence = self._engine_defaults(engine)
            ts = now()
            prompt = (voice_description_prompt or "").strip() or None
            v = VoiceRecord(voice_id=new_id(), voice_name=name, kind=kind, engine=engine, gender=gender,
                            description=description.strip(), catalog_slot=catalog_slot,
                            voice_description_prompt=prompt,
                            prompt_version=1 if prompt else 0,
                            prompt_versions=[VoicePromptVersion(version=1, prompt=prompt, created_at=ts)] if prompt else [],
                            sample_text=(sample_text or DEFAULT_AUDITION_TEXT).strip(),
                            generation_parameters=generation_parameters or {}, seed=seed,
                            reference_transcript=(reference_transcript or "").strip() or None,
                            model_id=model_id, model_revision=revision, license_metadata=licence,
                            origin=origin or {}, created_at=ts, updated_at=ts)
            self.voice_dir(v.voice_id).mkdir(parents=True, exist_ok=True)
            return self._save(v)

    def update(self, voice_id: str, **changes) -> VoiceRecord:
        with self._lock:
            v = self.get(voice_id)
            if "voice_name" in changes and changes["voice_name"] is not None:
                changes["voice_name"] = self._check_name(changes["voice_name"], exclude=voice_id)
            prompt = changes.get("voice_description_prompt")
            if prompt is not None:
                prompt = prompt.strip()
                if engine_class(v.engine).get_capabilities().prompt_conditioned and not prompt:
                    raise ValidationFailed(f"{v.engine.value} voices need a voice_description_prompt")
                if prompt != v.voice_description_prompt:
                    version = v.prompt_version + 1
                    changes["voice_description_prompt"] = prompt or None
                    changes["prompt_version"] = version
                    changes["prompt_versions"] = [*v.prompt_versions, VoicePromptVersion(
                        version=version, prompt=prompt, created_at=now())]
            before = v.config_hash()
            data = v.model_dump() | {k: val for k, val in changes.items() if val is not None} | {"updated_at": now()}
            v = VoiceRecord.model_validate(data)
            if v.config_hash() != before and v.approved:
                v.approved = False  # identity-relevant change: must be auditioned again
                v.approval_note = "Configuration changed after approval; audition and approve again."
            return self._save(v)

    def delete(self, voice_id: str, in_use: Callable[[str], list[str]]) -> None:
        self.get(voice_id)
        users = in_use(voice_id)
        if users:
            raise ConflictError(f"Voice is assigned to {len(users)} character voice profile(s)/open job(s); unassign first.")
        with self._lock:
            shutil.rmtree(self.voice_dir(voice_id), ignore_errors=True)

    def set_reference(self, voice_id: str, src: Path, transcript: Optional[str], replace: bool,
                      origin: Optional[dict] = None) -> tuple[VoiceRecord, ReferenceReport]:
        """Validate + normalise ``src`` as the master reference. Approved masters need ``replace=True``."""
        with self._lock:
            v = self.get(voice_id)
            caps = engine_class(v.engine).get_capabilities()
            if not caps.reference_conditioned:
                raise ValidationFailed(f"{v.engine.value} voices are prompt-conditioned and take no reference audio")
            if caps.needs_reference_transcript and not (transcript or v.reference_transcript or "").strip():
                raise ValidationFailed(f"{v.engine.value} needs the transcript of the reference recording")
            d = self.voice_dir(voice_id)
            final = d / "reference.wav"
            if final.is_file() and not replace:
                raise ConflictError("This voice already has a master reference; confirm replacement to overwrite it.")
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
            sha = hashlib.sha256(final.read_bytes()).hexdigest()
            return self.update(voice_id, reference_available=True, reference_duration_seconds=report.duration_seconds,
                               reference_speech_seconds=report.speech_seconds, reference_sample_rate=report.sample_rate,
                               reference_quality=list(report.warnings), reference_sha256=sha,
                               reference_transcript=(transcript or v.reference_transcript or "").strip() or None,
                               origin={**v.origin, **(origin or {})}), report

    # ------------------------------------------------------------------ catalog
    def seed_catalog(self) -> tuple[list[VoiceRecord], list[VoiceRecord]]:
        """Create or refresh the curated Parler slots; prompt changes invalidate prior approval."""
        existing = {v.catalog_slot: v for v in self.list() if v.catalog_slot}
        created: list[VoiceRecord] = []
        updated: list[VoiceRecord] = []
        for slot in CATALOG_SLOTS:
            current = existing.get(slot["slot"])
            if current:
                if not current.origin.get("catalog"):
                    continue
                changes = {
                    "voice_name": slot["voice_name"],
                    "description": slot["description"] + " (catalog slot; audition and approve before use)",
                    "voice_description_prompt": slot["prompt"],
                }
                if any(getattr(current, key) != value for key, value in changes.items()):
                    updated.append(self.update(current.voice_id, **changes))
                continue
            v = self.create(voice_name=slot["voice_name"], engine=TTSEngine.PARLER, gender=slot["gender"],
                            description=slot["description"] + " (catalog slot; audition and approve before use)",
                            voice_description_prompt=slot["prompt"], catalog_slot=slot["slot"],
                            origin={"catalog": True})
            created.append(self.update(v.voice_id, active=False))
        return created, updated

    # ------------------------------------------------------------------ audition / approval
    def to_spec(self, v: VoiceRecord) -> VoiceSpec:
        ref = self.reference_path(v.voice_id)
        return VoiceSpec(engine=v.engine, reference_path=ref if v.reference_available and ref.is_file() else None,
                         reference_transcript=v.reference_transcript, voice_prompt=v.voice_description_prompt,
                         generation_parameters=dict(v.generation_parameters), seed=v.seed, voice_id=v.voice_id,
                         label=v.voice_name)

    def record_run(self, voice_id: str, run: dict) -> None:
        with open(self.voice_dir(voice_id) / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(run, ensure_ascii=False, default=str) + "\n")

    def store_preview(self, voice_id: str, wav_path: Path, text: str, metadata: dict, duration: float,
                      problems: list[str], run_id: str) -> VoiceRecord:
        with self._lock:
            v = self.get(voice_id)
            artifact = self.audition_path(voice_id, run_id)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(wav_path), str(artifact))
            shutil.copy2(artifact, self.preview_path(voice_id))
            preview = VoicePreview(text=text, generated_at=now(), config_hash=v.config_hash(), engine=v.engine,
                                   prompt_version=v.prompt_version, artifact_name=artifact.relative_to(
                                       self.voice_dir(voice_id)).as_posix(),
                                   model_id=metadata.get("model_id", v.model_id),
                                   model_revision=metadata.get("model_revision", v.model_revision),
                                   duration_seconds=round(duration, 3), run_id=run_id, problems=problems)
            self.record_run(voice_id, {"run_id": run_id, "kind": "audition", "text": text, **metadata,
                                       "config_hash": preview.config_hash, "prompt_version": v.prompt_version,
                                       "artifact_name": preview.artifact_name, "problems": problems,
                                       "at": preview.generated_at})
            return self._save(v.model_copy(update={"preview": preview, "updated_at": now()}))

    def set_approval(self, voice_id: str, approved: bool, note: str = "", benchmark: Optional[dict] = None) -> VoiceRecord:
        with self._lock:
            v = self.get(voice_id)
            if approved:
                if v.preview is None or v.preview.config_hash != v.config_hash():
                    raise ConflictError("Audition this voice with its current configuration before approving it.")
                if v.preview.problems:
                    raise ConflictError(f"The last audition reported problems: {'; '.join(v.preview.problems)}")
                changes = {"approved": True, "approved_at": now(), "approved_config_hash": v.config_hash(),
                           "approval_note": note.strip(), "active": True,
                           "supported_language_status": "validated"}
            else:
                changes = {"approved": False, "approved_at": None, "approved_config_hash": None,
                           "approval_note": note.strip(),
                           "supported_language_status": "rejected" if note.strip() else v.supported_language_status}
            if benchmark is not None:
                changes["quality_benchmark"] = benchmark
            return self._save(v.model_copy(update={**changes, "updated_at": now()}))

    # ------------------------------------------------------------------ resolution
    def lookup(self, voice_id: str) -> Optional[tuple[VoiceSpec, VoiceRecord]]:
        """Generation-time resolver: approved + active voices only; never substitutes."""
        try:
            v = self.get(voice_id)
        except NotFoundError:
            return None
        if not v.selectable:
            return None
        if v.engine not in self.s.enabled_tts_engines:
            return None
        spec = self.to_spec(v)
        if engine_class(v.engine).get_capabilities().reference_conditioned and spec.reference_path is None:
            return None
        return spec, v
