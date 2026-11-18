# hindi-dubbing-platform/backend/services/draft.py
"""Editable draft manifest for the human review gate.

Lifecycle:  analysis -> ``build_draft`` -> draft.json (``review_status=draft``, ``version`` 1)
            -> PATCH (``apply_patch``, optimistic ``base_version`` check, version+1 per save)
            -> confirm (``validate_for_confirm``; immutable ``approved_v<version>.json`` snapshot)
            -> ``to_generation_plan`` -> pipeline generation.

Speaker ids in the draft start as the diarization labels (``SPEAKER_00``). Renaming changes only the
display name; merging moves every segment of one speaker to another and records the diarization
labels it absorbed (``origin_labels``), which keeps clone-reference statistics and the clean-speech
selection correct after edits. ``speakers`` is the source of truth for voice selection; the voice
fields on each segment are a read-only denormalised view for clients.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from backend.app.config import TARGET_LANGUAGE, MixMode, Settings, TTSEngine
from backend.services.storage import SPEAKER_LABEL_RE, ConflictError, ValidationFailed, now
from backend.services.voices import DefaultVoiceBank, VoiceRequest, VoiceSource

DRAFT_SCHEMA_VERSION = 1
PIPELINE_VERSION = "2.0"
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
MIN_SEGMENT_SECONDS = 0.2


class SegmentReviewStatus(str, Enum):
    PENDING = "pending"
    EDITED = "edited"
    APPROVED = "approved"


class DraftSegment(BaseModel):
    segment_id: str
    start: float
    end: float
    original_text: str
    translated_text: str
    machine_translation: str
    speaker_id: str
    confidence: Optional[float] = None
    speaker_confidence: Optional[float] = None   # v3: diarization attribution confidence (0..1)
    review_status: SegmentReviewStatus = SegmentReviewStatus.PENDING
    warnings: list[str] = Field(default_factory=list)
    # read-only view (filled by ``present``)
    speaker_display_name: Optional[str] = None
    voice_source: Optional[VoiceSource] = None
    default_voice_id: Optional[str] = None
    library_speaker_id: Optional[str] = None
    reference_duration_seconds: Optional[float] = None


class DraftSpeaker(BaseModel):
    speaker_id: str
    display_name: str
    color: int = 0
    origin_labels: list[str] = Field(default_factory=list)
    segment_count: int = 0
    total_speech_duration: float = 0.0
    usable_reference_duration: float = 0.0
    reference_available: bool = False
    voice_source: VoiceSource = VoiceSource.CLONE
    library_speaker_id: Optional[str] = None
    default_voice_id: Optional[str] = None
    available_voice_options: list[VoiceSource] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # v3: engine / Voice Library / canonical character (all optional; v2 drafts load unchanged)
    engine: Optional[TTSEngine] = None            # clone engine (IndicF5); view for other sources
    voice_id: Optional[str] = None                # cast: Voice Library voice
    character_id: Optional[str] = None            # user-confirmed canonical character
    character_name: Optional[str] = None          # read-only view
    mapping_status: str = "none"                  # none | unassigned | suggested | review | confirmed | rejected
    suggested_character_id: Optional[str] = None
    suggested_character_name: Optional[str] = None
    identity_confidence: Optional[float] = None
    identity_reason: str = ""
    identity_candidates: list[dict] = Field(default_factory=list)
    previous_episode_matches: list[dict] = Field(default_factory=list)
    reference_quality: list[str] = Field(default_factory=list)
    available_engines: list[TTSEngine] = Field(default_factory=list)


class DraftManifest(BaseModel):
    manifest_version: int = DRAFT_SCHEMA_VERSION
    version: int = 1
    job_id: str
    project_id: str
    video_id: str
    source_language: str
    target_language: str = TARGET_LANGUAGE
    pipeline_version: str = PIPELINE_VERSION
    media_duration: float
    mix_mode: MixMode = MixMode.PRESERVE_BACKGROUND
    created_at: datetime
    updated_at: datetime
    review_status: Literal["draft", "approved"] = "draft"
    approved_version: Optional[int] = None
    approved_at: Optional[datetime] = None
    segments: list[DraftSegment]
    speakers: list[DraftSpeaker]
    warnings: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------------------- patch schema
class SegmentPatch(BaseModel):
    segment_id: str
    translated_text: Optional[str] = None
    original_text: Optional[str] = None
    speaker_id: Optional[str] = None
    start: Optional[float] = Field(None, ge=0)
    end: Optional[float] = Field(None, ge=0)
    review_status: Optional[SegmentReviewStatus] = None


class SpeakerPatch(BaseModel):
    speaker_id: str
    display_name: Optional[str] = Field(None, min_length=1, max_length=80)
    voice_source: Optional[VoiceSource] = None
    library_speaker_id: Optional[str] = None
    default_voice_id: Optional[str] = None
    # v3
    engine: Optional[TTSEngine] = None
    voice_id: Optional[str] = None
    character_id: Optional[str] = None            # set -> confirmed mapping; "" -> clear
    reject_suggestion: Optional[bool] = None


class NewSpeaker(BaseModel):
    speaker_id: str
    display_name: str = Field(..., min_length=1, max_length=80)

    @field_validator("speaker_id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not SPEAKER_LABEL_RE.fullmatch(v):
            raise ValueError("speaker_id must be 1-64 chars of letters, digits, '_' or '-'")
        return v


class MergeOp(BaseModel):
    source_speaker_id: str
    target_speaker_id: str


class DraftPatch(BaseModel):
    base_version: int = Field(..., ge=1)
    new_speakers: list[NewSpeaker] = Field(default_factory=list)
    merges: list[MergeOp] = Field(default_factory=list)
    speakers: list[SpeakerPatch] = Field(default_factory=list)
    segments: list[SegmentPatch] = Field(default_factory=list)
    mix_mode: Optional[MixMode] = None


# ------------------------------------------------------------------------------- build
def build_draft(*, job_id: str, project_id: str, video_id: str, analysis, mix_mode: MixMode,
                settings: Settings, bank: DefaultVoiceBank) -> DraftManifest:
    """Create the first draft from an ``AnalysisResult``."""
    words = [w for s in analysis.transcript.get("segments", []) for w in (s.get("words") or [])]
    segments = []
    for b in analysis.blocks:
        probs = [w["probability"] for w in words
                 if "probability" in w and w["start"] < b["end"] and w["end"] > b["start"]]
        text = analysis.translations.get(b["id"], "")
        segments.append(DraftSegment(
            segment_id=b["id"], start=b["start"], end=b["end"], original_text=b["text"], translated_text=text,
            machine_translation=text, speaker_id=b["speaker"],
            confidence=round(sum(probs) / len(probs), 3) if probs else None,
            speaker_confidence=b.get("speaker_confidence")))
    labels = sorted({b["speaker"] for b in analysis.blocks})
    speakers = []
    for i, label in enumerate(labels):
        prof = analysis.profiles.get(label) or {}
        usable = float(prof.get("available_seconds", 0.0))
        spk = DraftSpeaker(speaker_id=label, display_name=f"Speaker {i + 1}", color=i, origin_labels=[label],
                           usable_reference_duration=round(usable, 2), reference_available=bool(prof.get("reference")),
                           voice_source=VoiceSource.CAST, engine=TTSEngine.PARLER)
        speakers.append(spk)
    ts = now()
    draft = DraftManifest(job_id=job_id, project_id=project_id, video_id=video_id,
                          source_language=analysis.source_language, media_duration=analysis.media.info.duration,
                          mix_mode=mix_mode, created_at=ts, updated_at=ts, segments=segments, speakers=speakers,
                          warnings=list(analysis.warnings))
    return refresh(draft, analysis.profiles, settings)


def refresh(draft: DraftManifest, profiles: dict, settings: Settings) -> DraftManifest:
    """Recompute derived statistics and warnings after any edit."""
    by_spk: dict[str, list[DraftSegment]] = {s.speaker_id: [] for s in draft.speakers}
    for seg in draft.segments:
        by_spk.setdefault(seg.speaker_id, []).append(seg)
    minimum = settings.min_speaker_reference_seconds
    for spk in draft.speakers:
        segs = by_spk.get(spk.speaker_id, [])
        spk.segment_count = len(segs)
        spk.total_speech_duration = round(sum(s.end - s.start for s in segs), 2)
        spk.usable_reference_duration = round(
            sum(float((profiles.get(lbl) or {}).get("available_seconds", 0.0)) for lbl in spk.origin_labels), 2)
        spk.reference_available = any((profiles.get(lbl) or {}).get("reference") for lbl in spk.origin_labels)
        options = [VoiceSource.LIBRARY, VoiceSource.CAST, VoiceSource.CHARACTER]
        if spk.reference_available:
            options.insert(0, VoiceSource.CLONE)
        spk.available_voice_options = options
        spk.available_engines = sorted(settings.enabled_tts_engines, key=lambda e: e.value)
        spk.reference_quality = [w for lbl in spk.origin_labels for w in ((profiles.get(lbl) or {}).get("quality") or [])]
        if spk.voice_source == VoiceSource.CAST and not spk.voice_id:
            spk.warnings.append("Select an approved Parler voice profile before generating.")
        spk.warnings = []
        if spk.voice_source == VoiceSource.CLONE and (spk.engine is None or spk.engine == TTSEngine.XTTS):
            spk.warnings.append("This saved clone assignment refers to retired XTTS or has no engine; select IndicF5 "
                                "with a transcript-bearing reference, or choose an approved Parler profile.")
        if spk.voice_source == VoiceSource.CLONE and spk.engine == TTSEngine.INDICF5 and \
                not any((profiles.get(lbl) or {}).get("transcript") for lbl in spk.origin_labels):
            spk.warnings.append("IndicF5 cloning needs a reference transcript; this video's clone reference has none. "
                                "Choose a transcript-bearing library reference or an approved Parler profile.")
        if spk.mapping_status in ("suggested", "review") and spk.suggested_character_name:
            spk.warnings.append(f"Suggested character: {spk.suggested_character_name} "
                                f"({(spk.identity_confidence or 0) * 100:.0f}% similar); confirm or reject.")
        elif spk.mapping_status == "unassigned" and spk.identity_reason:
            spk.warnings.append(f"Character unassigned: {spk.identity_reason}")
        if spk.voice_source == VoiceSource.CLONE and spk.usable_reference_duration < minimum:
            spk.warnings.append(
                f"Only {spk.usable_reference_duration:.1f}s of clean speech (minimum {minimum:g}s): "
                "the clone may be less faithful; consider a library reference or approved Parler profile.")
        if spk.segment_count == 0:
            spk.warnings.append("No segments are assigned to this speaker.")
    ordered = sorted(draft.segments, key=lambda s: s.start)
    for i, seg in enumerate(ordered):
        seg.warnings = []
        if not seg.translated_text.strip():
            seg.warnings.append("Hindi translation is empty.")
        elif not _DEVANAGARI.search(seg.translated_text):
            seg.warnings.append("Hindi translation contains no Devanagari text.")
        if seg.confidence is not None and seg.confidence < 0.6:
            seg.warnings.append(f"Low ASR confidence ({seg.confidence:.2f}); check the original text.")
        if seg.end - seg.start < settings.min_segment_identity_duration:
            seg.warnings.append("Short line: speaker attribution is uncertain; verify the speaker.")
        if seg.speaker_confidence is not None and seg.speaker_confidence < 0.5:
            seg.warnings.append(f"Low speaker-attribution confidence ({seg.speaker_confidence:.2f}).")
        if i + 1 < len(ordered) and ordered[i + 1].start < seg.end - 0.05 and ordered[i + 1].speaker_id != seg.speaker_id:
            seg.warnings.append("Overlaps the next line of another speaker.")
    draft.segments = ordered
    return draft


# ------------------------------------------------------------------------------- edit
def apply_patch(draft: DraftManifest, patch: DraftPatch, profiles: dict, settings: Settings) -> DraftManifest:
    if draft.review_status != "draft":
        raise ConflictError("Draft is approved and locked; it can no longer be edited.")
    if patch.base_version != draft.version:
        raise ConflictError(f"Draft was modified (current version {draft.version}, yours {patch.base_version}); "
                            "reload before saving.")
    d = draft.model_copy(deep=True)
    errors: list[str] = []
    speakers = {s.speaker_id: s for s in d.speakers}

    for ns in patch.new_speakers:
        if ns.speaker_id in speakers:
            errors.append(f"Speaker id {ns.speaker_id} already exists.")
            continue
        spk = DraftSpeaker(speaker_id=ns.speaker_id, display_name=ns.display_name.strip(),
                           color=max((s.color for s in d.speakers), default=-1) + 1,
                           voice_source=VoiceSource.CAST, engine=TTSEngine.PARLER)
        d.speakers.append(spk)
        speakers[spk.speaker_id] = spk

    for m in patch.merges:
        src, dst = speakers.get(m.source_speaker_id), speakers.get(m.target_speaker_id)
        if not src or not dst or src is dst:
            errors.append(f"Invalid merge {m.source_speaker_id} -> {m.target_speaker_id}.")
            continue
        for seg in d.segments:
            if seg.speaker_id == src.speaker_id:
                seg.speaker_id = dst.speaker_id
        dst.origin_labels = sorted(set(dst.origin_labels) | set(src.origin_labels))
        d.speakers = [s for s in d.speakers if s is not src]
        del speakers[src.speaker_id]

    for sp in patch.speakers:
        spk = speakers.get(sp.speaker_id)
        if spk is None:
            errors.append(f"Unknown speaker {sp.speaker_id}.")
            continue
        if sp.display_name is not None:
            spk.display_name = sp.display_name.strip()
        if sp.voice_source is not None:
            spk.voice_source = sp.voice_source
        if "library_speaker_id" in sp.model_fields_set:
            spk.library_speaker_id = sp.library_speaker_id or None
        if "default_voice_id" in sp.model_fields_set:
            spk.default_voice_id = sp.default_voice_id or None
        if sp.engine is not None:
            spk.engine = sp.engine
        if "voice_id" in sp.model_fields_set:
            spk.voice_id = sp.voice_id or None
        if "character_id" in sp.model_fields_set:
            if sp.character_id:
                spk.character_id, spk.mapping_status = sp.character_id, "confirmed"
                if spk.suggested_character_id == sp.character_id:
                    spk.identity_reason = "suggestion confirmed by reviewer"
            else:
                spk.character_id, spk.character_name = None, None
                spk.mapping_status = "unassigned" if spk.suggested_character_id is None else "rejected"
        if sp.reject_suggestion:
            spk.mapping_status = "rejected"
            spk.warnings = [w for w in spk.warnings if not w.startswith("Suggested character")]

    names = [s.display_name.casefold() for s in d.speakers]
    if len(names) != len(set(names)):
        errors.append("Speaker display names must be unique within a video.")

    segs = {s.segment_id: s for s in d.segments}
    for sp in patch.segments:
        seg = segs.get(sp.segment_id)
        if seg is None:
            errors.append(f"Unknown segment {sp.segment_id}.")
            continue
        if sp.speaker_id is not None:
            if sp.speaker_id not in speakers:
                errors.append(f"{sp.segment_id}: unknown speaker {sp.speaker_id}.")
            else:
                seg.speaker_id = sp.speaker_id
        if sp.translated_text is not None:
            seg.translated_text = sp.translated_text.strip()
        if sp.original_text is not None:
            seg.original_text = sp.original_text.strip()
        start = sp.start if sp.start is not None else seg.start
        end = sp.end if sp.end is not None else seg.end
        if end - start < MIN_SEGMENT_SECONDS or end > d.media_duration + 0.01:
            errors.append(f"{sp.segment_id}: invalid timing {start:.2f}-{end:.2f}s "
                          f"(min {MIN_SEGMENT_SECONDS}s, within {d.media_duration:.2f}s).")
        else:
            seg.start, seg.end = round(start, 3), round(end, 3)
        if sp.review_status is not None:
            seg.review_status = sp.review_status
        elif sp.translated_text is not None and seg.translated_text != seg.machine_translation:
            seg.review_status = SegmentReviewStatus.EDITED

    if errors:
        raise ValidationFailed("Draft update rejected", errors)
    if patch.mix_mode is not None:
        d.mix_mode = patch.mix_mode
    d.version += 1
    d.updated_at = now()
    return refresh(d, profiles, settings)


# ------------------------------------------------------------------------------- confirm
def validate_for_confirm(draft: DraftManifest, library_ids: set[str], bank: DefaultVoiceBank,
                         settings: Settings, cast_ids: Optional[set[str]] = None,
                         character_ids: Optional[set[str]] = None,
                         profiles: Optional[dict] = None) -> tuple[list[str], list[str]]:
    """(blocking errors, warnings) for confirmation.

    ``cast_ids``: Voice Library voices that are approved + active; ``character_ids``: characters with an
    approved active voice profile. Both default to *empty*, so a v2 caller cannot accidentally pass.
    """
    errors, warnings = [], []
    cast_ids, character_ids, profiles = cast_ids or set(), character_ids or set(), profiles or {}
    speakers = {s.speaker_id: s for s in draft.speakers}
    if not draft.segments:
        errors.append("The draft has no dialogue segments.")
    for seg in draft.segments:
        if seg.speaker_id not in speakers:
            errors.append(f"{seg.segment_id}: speaker {seg.speaker_id} does not exist.")
        if not seg.translated_text.strip():
            errors.append(f"{seg.segment_id}: Hindi translation is empty.")
        elif not _DEVANAGARI.search(seg.translated_text):
            errors.append(f"{seg.segment_id}: Hindi translation must be written in Devanagari.")
        if seg.end - seg.start < MIN_SEGMENT_SECONDS or seg.start < 0 or seg.end > draft.media_duration + 0.01:
            errors.append(f"{seg.segment_id}: invalid timestamps {seg.start:.2f}-{seg.end:.2f}s.")
    used = {seg.speaker_id for seg in draft.segments}
    for spk in draft.speakers:
        if spk.speaker_id not in used:
            continue
        if spk.voice_source == VoiceSource.LIBRARY:
            if not spk.library_speaker_id or spk.library_speaker_id not in library_ids:
                errors.append(f"{spk.display_name}: select a project library voice with a reference recording and transcript.")
        elif spk.voice_source == VoiceSource.DEFAULT:
            errors.append(f"{spk.display_name}: saved default voices used retired XTTS; choose a supported voice.")
        elif spk.voice_source == VoiceSource.CAST:
            if not spk.voice_id or spk.voice_id not in cast_ids:
                errors.append(f"{spk.display_name}: select an approved, active Voice Library voice.")
        elif spk.voice_source == VoiceSource.CHARACTER:
            if not spk.character_id:
                errors.append(f"{spk.display_name}: assign a canonical character before using its voice.")
            elif spk.character_id not in character_ids:
                errors.append(f"{spk.display_name}: the assigned character has no approved active voice profile.")
        else:
            engine = spk.engine
            if engine is None or engine == TTSEngine.XTTS:
                errors.append(f"{spk.display_name}: saved clone assignment refers to retired XTTS or has no engine; "
                              "explicitly select IndicF5 or an approved Parler profile.")
                warnings.extend(spk.warnings)
                continue
            if engine != TTSEngine.INDICF5:
                errors.append(f"{spk.display_name}: clone engine '{engine.value}' is unsupported; choose IndicF5 or "
                              "an approved Parler profile.")
            if engine not in settings.enabled_tts_engines:
                errors.append(f"{spk.display_name}: clone engine '{engine.value}' is not enabled.")
            if engine == TTSEngine.INDICF5 and \
                    not any((profiles.get(lbl) or {}).get("transcript") for lbl in spk.origin_labels):
                errors.append(f"{spk.display_name}: IndicF5 cloning needs a reference transcript (not available for "
                              "this video); choose a transcript-bearing project library reference or Parler profile.")
            if not spk.reference_available:
                errors.append(f"{spk.display_name}: no usable speech to clone; choose a project library reference "
                              "or approved Parler profile.")
            warnings.extend(spk.warnings)
        if spk.mapping_status in ("suggested", "review"):
            warnings.append(f"{spk.display_name}: character suggestion not confirmed; this episode's speaker will not "
                            "be linked to a canonical character.")
    return errors, warnings


def to_generation_inputs(draft: DraftManifest) -> tuple[list[dict], dict[str, str], dict[str, VoiceRequest], set[str], dict[str, str]]:
    """(blocks, translations, voice requests, locked ids, diarization-label -> final speaker map)."""
    blocks = [{"id": s.segment_id, "start": s.start, "end": s.end, "speaker": s.speaker_id, "text": s.original_text}
              for s in sorted(draft.segments, key=lambda s: s.start)]
    translations = {s.segment_id: s.translated_text for s in draft.segments}
    locked = {s.segment_id for s in draft.segments
              if s.translated_text != s.machine_translation or s.review_status == SegmentReviewStatus.APPROVED}
    requests = {s.speaker_id: VoiceRequest(speaker_id=s.speaker_id, display_name=s.display_name,
                                           voice_source=s.voice_source, library_speaker_id=s.library_speaker_id,
                                           default_voice_id=s.default_voice_id, engine=s.engine, voice_id=s.voice_id,
                                           character_id=s.character_id) for s in draft.speakers}
    label_map = {lbl: s.speaker_id for s in draft.speakers for lbl in s.origin_labels}
    return blocks, translations, requests, locked, label_map


def attach_identity(draft: DraftManifest, suggestions: list[dict], character_names: dict[str, str],
                    history: dict[str, list[dict]]) -> DraftManifest:
    """Write cross-episode suggestions onto speakers that the reviewer has not already decided."""
    by_label = {s["local_speaker_id"]: s for s in suggestions}
    for spk in draft.speakers:
        if spk.mapping_status in ("confirmed", "rejected"):
            spk.character_name = character_names.get(spk.character_id or "", spk.character_name)
            continue
        hits = [by_label[lbl] for lbl in spk.origin_labels if lbl in by_label]
        if not hits:
            continue
        best = max(hits, key=lambda h: (h.get("confidence") or -1.0, h.get("speech_seconds") or 0.0))
        spk.mapping_status = best["status"]
        spk.suggested_character_id = best.get("character_id") if best["status"] != "unassigned" else None
        spk.suggested_character_name = character_names.get(spk.suggested_character_id or "")
        spk.identity_confidence = best.get("confidence")
        spk.identity_reason = best.get("reason", "")
        spk.identity_candidates = [{**c, "character_name": character_names.get(c["character_id"], "")}
                                   for c in best.get("candidates", [])]
        spk.previous_episode_matches = history.get(spk.suggested_character_id or "", [])
    return draft


def present(draft: DraftManifest) -> DraftManifest:
    """Copy with the read-only per-segment voice view filled in from the speaker table."""
    d = draft.model_copy(deep=True)
    speakers = {s.speaker_id: s for s in d.speakers}
    for seg in d.segments:
        spk = speakers.get(seg.speaker_id)
        if spk:
            seg.speaker_display_name = spk.display_name
            seg.voice_source = spk.voice_source
            seg.default_voice_id = spk.default_voice_id
            seg.library_speaker_id = spk.library_speaker_id
            seg.reference_duration_seconds = spk.usable_reference_duration
    return d
