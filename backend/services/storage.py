# hindi-dubbing-platform/backend/services/storage.py

"""Small persistence helpers shared by the project store, drafts and the job manager.

Every JSON document is written to a temporary sibling and moved into place with ``os.replace``,
so a crash never leaves a half-written record. Identifiers that become path components are
validated against strict patterns: no client-supplied string is ever joined into a path unchecked.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ENTITY_ID_RE = re.compile(r"^[a-f0-9]{32}$")              # project / video / library-speaker / job ids
VOICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")  # default voice ids (file stems)
SPEAKER_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")  # per-video speaker ids (SPEAKER_00)
SEGMENT_ID_RE = re.compile(r"^seg_\d{5}$")


class NotFoundError(LookupError):
    pass


class ConflictError(RuntimeError):
    pass


class ValidationFailed(ValueError):
    def __init__(self, message: str, details: list[str] | None = None):
        super().__init__(message)
        self.details = details or []


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> datetime:
    return datetime.now(timezone.utc)


def is_entity_id(value: str) -> bool:
    return bool(ENTITY_ID_RE.fullmatch(value or ""))


def require_entity_id(value: str, kind: str) -> str:
    if not is_entity_id(value):
        raise NotFoundError(f"{kind} not found")
    return value


def atomic_write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        text = data.model_dump_json(indent=2)
    else:
        text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    tmp = path.with_name(f"{path.name}.part-{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def write_once_json(path: Path, data: Any) -> Path:
    """Create ``path`` exclusively; raises ``ConflictError`` if it already exists (immutable snapshots)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = data.model_dump_json(indent=2) if isinstance(data, BaseModel) else json.dumps(data, ensure_ascii=False, indent=2)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ConflictError(f"{path.name} already exists and is immutable") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
