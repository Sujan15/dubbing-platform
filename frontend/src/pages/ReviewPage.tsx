import { forwardRef, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { characterApi, draftApi, jobApi, speakerApi, urls, voiceApi } from "../api";
import { ApiError, errorText } from "../api/client";
import type {
  ConfirmPreview,
  DraftManifest,
  DraftPatch,
  DraftSegment,
  DraftSpeaker,
  LibrarySpeaker,
  Meta,
  SegmentPatch,
  SegmentReviewStatus,
  SeriesCharacter,
  SpeakerPatch,
  VoiceSource,
  VoiceProfile,
} from "../api/types";
import { useAsync } from "../hooks/useAsync";
import {
  Badge,
  Button,
  ErrorBanner,
  Modal,
  PlayButton,
  Spinner,
  VOICE_SOURCE_LABEL,
  WarningList,
  fmtTime,
  speakerColor,
} from "../components/ui";

interface Pending {
  segments: Record<string, SegmentPatch>;
  speakers: Record<string, SpeakerPatch>;
}
const EMPTY: Pending = { segments: {}, speakers: {} };

/** Draft with unsaved local edits applied, so the UI always shows what will be saved. */
export function applyPending(draft: DraftManifest, p: Pending): DraftManifest {
  const speakers = draft.speakers.map((s) => ({ ...s, ...stripUndefined(p.speakers[s.speaker_id]) }));
  const byId = new Map(speakers.map((s) => [s.speaker_id, s]));
  const segments = draft.segments.map((seg) => {
    const merged = { ...seg, ...stripUndefined(p.segments[seg.segment_id]) };
    const spk = byId.get(merged.speaker_id);
    return { ...merged, speaker_display_name: spk?.display_name ?? merged.speaker_id, voice_source: spk?.voice_source ?? null };
  });
  return { ...draft, speakers, segments };
}

function stripUndefined<T extends object>(o: T | undefined): Partial<T> {
  if (!o) return {};
  return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined)) as Partial<T>;
}

const isTyping = (t: EventTarget | null) =>
  t instanceof HTMLElement && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName));

export default function ReviewPage({ meta }: { meta?: Meta }) {
  const { jobId = "" } = useParams();
  const navigate = useNavigate();
  const job = useAsync((s) => jobApi.get(jobId, s), [jobId]);
  const draftQ = useAsync((s) => draftApi.get(jobId, s), [jobId]);
  const projectId = draftQ.data?.project_id;
  const library = useAsync<LibrarySpeaker[]>(() => (projectId ? speakerApi.list(projectId) : Promise.resolve([])), [projectId]);
  const castProfiles = useAsync(
    (s) => voiceApi.list({ approved: true, include_inactive: false, include_legacy: false }, s),
    [],
  );
  const characters = useAsync<SeriesCharacter[]>(
    (s) => (projectId ? characterApi.list(projectId, s) : Promise.resolve([])),
    [projectId],
  );
  const engines = useAsync((s) => voiceApi.engines(s), []);

  const [pending, setPending] = useState<Pending>(EMPTY);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<unknown>(null);
  const [conflict, setConflict] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [time, setTime] = useState(0);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [filterSpeaker, setFilterSpeaker] = useState<string>("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const rowRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const dirtyCount = Object.keys(pending.segments).length + Object.keys(pending.speakers).length;
  const draft = draftQ.data;
  const view = useMemo(() => (draft ? applyPending(draft, pending) : undefined), [draft, pending]);
  const readOnly = !draft || draft.review_status !== "draft" || job.data?.state !== "awaiting_review";

  // warn before leaving with unsaved edits
  useEffect(() => {
    if (!dirtyCount) return;
    const h = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", h);
    return () => window.removeEventListener("beforeunload", h);
  }, [dirtyCount]);

  const save = useCallback(
    async (extra: Partial<DraftPatch> = {}): Promise<DraftManifest | null> => {
      if (!draft) return null;
      setSaving(true);
      setSaveError(null);
      const patch: DraftPatch = {
        base_version: draft.version,
        segments: Object.values(pending.segments),
        speakers: Object.values(pending.speakers),
        ...extra,
      };
      try {
        const next = await draftApi.patch(jobId, patch);
        draftQ.setData(next);
        setPending(EMPTY);
        return next;
      } catch (e) {
        if (e instanceof ApiError && e.status === 409) setConflict(true);
        setSaveError(e);
        return null;
      } finally {
        setSaving(false);
      }
    },
    [draft, pending, jobId, draftQ],
  );

  const editSegment = (id: string, change: Omit<SegmentPatch, "segment_id">) =>
    setPending((p) => ({ ...p, segments: { ...p.segments, [id]: { ...p.segments[id], ...change, segment_id: id } } }));
  const editSpeaker = (id: string, change: Partial<SpeakerPatch>) =>
    setPending((p) => ({ ...p, speakers: { ...p.speakers, [id]: { ...p.speakers[id], ...change, speaker_id: id } } }));

  const segments = view?.segments ?? [];
  const visible = filterSpeaker ? segments.filter((s) => s.speaker_id === filterSpeaker) : segments;
  const speakerIndex = useMemo(() => new Map((view?.speakers ?? []).map((s) => [s.speaker_id, s])), [view]);

  const seek = useCallback((seg: DraftSegment, play = false) => {
    const v = videoRef.current;
    setActiveId(seg.segment_id);
    if (!v) return;
    v.currentTime = seg.start;
    if (play) void v.play();
  }, []);

  // follow playback
  const onTime = () => {
    const v = videoRef.current;
    if (!v) return;
    setTime(v.currentTime);
    const hit = segments.find((s) => v.currentTime >= s.start && v.currentTime < s.end);
    if (hit && hit.segment_id !== activeId) {
      setActiveId(hit.segment_id);
      if (!v.paused) rowRefs.current[hit.segment_id]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  };

  // keyboard: Space play/pause, J previous, K next, Ctrl/Cmd+S save
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        if (dirtyCount && !readOnly) void save();
        return;
      }
      if (isTyping(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
      const v = videoRef.current;
      if (e.key === " " && v) {
        e.preventDefault();
        if (v.paused) void v.play();
        else v.pause();
      } else if (e.key === "j" || e.key === "k") {
        const idx = visible.findIndex((s) => s.segment_id === activeId);
        const next = e.key === "k" ? visible[Math.min(visible.length - 1, idx + 1)] : visible[Math.max(0, idx - 1)];
        if (next) {
          seek(next);
          rowRefs.current[next.segment_id]?.scrollIntoView({ block: "nearest" });
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [visible, activeId, seek, save, dirtyCount, readOnly]);

  if ((draftQ.loading && !draft) || (job.loading && !job.data)) return <Spinner label="Loading draft" />;
  if (!draft || !view) return <ErrorBanner error={draftQ.error ?? "Draft not available"} onRetry={draftQ.reload} />;

  const libraryVoices = library.data ?? [];
  const approvedProfiles = (castProfiles.data?.voices ?? []).filter((voice) => voice.selectable);
  const seriesCharacters = characters.data ?? [];
  const cloneEngine = engines.data?.default_clone_engine ?? "indicf5";
  const minRef = meta?.min_speaker_reference_seconds ?? 6;

  return (
    <section className="review">
      <div className="review-head">
        <div className="crumbs">
          <Link to="/">Projects</Link> / <Link to={`/projects/${draft.project_id}`}>Project</Link> /{" "}
          <span>{job.data?.input_filename}</span>
        </div>
        <div className="row gap">
          <span className="muted small">
            Draft v{draft.version} · {view.segments.length} lines · {view.speakers.length} speakers · source{" "}
            {draft.source_language}
          </span>
          {dirtyCount > 0 && <Badge tone="warn">{dirtyCount} unsaved change{dirtyCount > 1 ? "s" : ""}</Badge>}
          {!readOnly && (
            <>
              <Button onClick={() => save()} busy={saving} disabled={!dirtyCount} title="Ctrl+S">
                Save draft
              </Button>
              <Button
                variant="primary"
                onClick={async () => {
                  if (dirtyCount && !(await save())) return;
                  setConfirmOpen(true);
                }}
                disabled={saving}
              >
                Confirm & Generate
              </Button>
            </>
          )}
        </div>
      </div>

      {readOnly && (
        <div className="banner banner-info">
          {draft.review_status === "approved"
            ? `Approved draft v${draft.approved_version} is locked.`
            : `Job is ${job.data?.state}; the draft is read-only.`}{" "}
          <Link to={`/jobs/${jobId}`}>View job</Link>
        </div>
      )}
      {conflict && (
        <div className="banner banner-error" role="alert">
          This draft was changed elsewhere. Reloading discards your unsaved edits.
          <Button
            size="sm"
            onClick={() => {
              setPending(EMPTY);
              setConflict(false);
              setSaveError(null);
              draftQ.reload();
            }}
          >
            Reload draft
          </Button>
        </div>
      )}
      {!conflict && <ErrorBanner error={saveError} />}
      <WarningList warnings={draft.warnings} title="Analysis warnings" />

      <div className="workstation">
        <div className="ws-left">
          <div className="player-wrap">
            <video
              ref={videoRef}
              className="player"
              src={urls.sourceVideo(jobId)}
              controls
              preload="metadata"
              onTimeUpdate={onTime}
            />
          </div>
          <Timeline
            duration={draft.media_duration}
            time={time}
            segments={segments}
            speakers={speakerIndex}
            activeId={activeId}
            onSelect={(s) => {
              seek(s);
              rowRefs.current[s.segment_id]?.scrollIntoView({ block: "nearest" });
            }}
            onSeek={(t) => {
              if (videoRef.current) videoRef.current.currentTime = t;
            }}
          />
          <p className="muted small kbd-help">
            <kbd>Space</kbd> play/pause · <kbd>J</kbd>/<kbd>K</kbd> previous/next line · <kbd>Ctrl</kbd>+<kbd>S</kbd> save
          </p>
          <SpeakerRail
            jobId={jobId}
            projectId={draft.project_id}
            speakers={view.speakers}
            library={libraryVoices}
            castProfiles={approvedProfiles}
            characters={seriesCharacters}
            cloneEngine={cloneEngine}
            minRef={minRef}
            readOnly={readOnly}
            saving={saving}
            filter={filterSpeaker}
            onFilter={setFilterSpeaker}
            onEdit={editSpeaker}
            onMerge={(source, target) => save({ merges: [{ source_speaker_id: source, target_speaker_id: target }] })}
            onAdd={(name) => {
              const ids = new Set(view.speakers.map((s) => s.speaker_id));
              let n = 1;
              while (ids.has(`NEW_${n}`)) n++;
              return save({ new_speakers: [{ speaker_id: `NEW_${n}`, display_name: name }] });
            }}
          />
        </div>

        <div className="ws-right" aria-label="Dialogue segments">
          <div className="seg-toolbar">
            <strong>Dialogue</strong>
            <select className="input input-sm" value={filterSpeaker} onChange={(e) => setFilterSpeaker(e.target.value)} aria-label="Filter by speaker">
              <option value="">All speakers</option>
              {view.speakers.map((s) => (
                <option key={s.speaker_id} value={s.speaker_id}>
                  {s.display_name}
                </option>
              ))}
            </select>
            <span className="muted small">
              {view.segments.filter((s) => s.warnings.length).length} with warnings ·{" "}
              {view.segments.filter((s) => s.review_status === "approved").length} approved
            </span>
          </div>
          <div className="seg-list">
            {visible.map((seg) => (
              <SegmentRow
                key={seg.segment_id}
                ref={(el) => (rowRefs.current[seg.segment_id] = el)}
                jobId={jobId}
                seg={seg}
                speakers={view.speakers}
                speaker={speakerIndex.get(seg.speaker_id)}
                active={seg.segment_id === activeId}
                dirty={!!pending.segments[seg.segment_id]}
                readOnly={readOnly}
                duration={draft.media_duration}
                onFocus={() => seek(seg)}
                onChange={(c) => editSegment(seg.segment_id, c)}
              />
            ))}
          </div>
        </div>
      </div>

      {confirmOpen && (
        <ConfirmDialog
          jobId={jobId}
          version={draft.version}
          library={libraryVoices}
          castProfiles={approvedProfiles}
          characters={seriesCharacters}
          onClose={() => setConfirmOpen(false)}
          onConfirmed={() => navigate(`/jobs/${jobId}`)}
        />
      )}
    </section>
  );
}

// ------------------------------------------------------------------------------- timeline
function Timeline(props: {
  duration: number;
  time: number;
  segments: DraftSegment[];
  speakers: Map<string, DraftSpeaker>;
  activeId: string | null;
  onSelect: (s: DraftSegment) => void;
  onSeek: (t: number) => void;
}) {
  const { duration, time, segments, speakers, activeId } = props;
  const rows = Array.from(speakers.values());
  const pct = (t: number) => `${(Math.max(0, Math.min(duration, t)) / duration) * 100}%`;
  return (
    <div className="timeline" aria-label="Timeline">
      <div className="tl-head">
        <span className="mono">{fmtTime(time, true)}</span>
        <span className="muted mono">{fmtTime(duration)}</span>
      </div>
      <div
        className="tl-body"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          props.onSeek(((e.clientX - r.left) / r.width) * duration);
        }}
      >
        {rows.map((spk) => (
          <div key={spk.speaker_id} className="tl-lane" title={spk.display_name}>
            {segments
              .filter((s) => s.speaker_id === spk.speaker_id)
              .map((s) => (
                <button
                  key={s.segment_id}
                  className={`tl-seg ${s.segment_id === activeId ? "active" : ""} ${s.warnings.length ? "warn" : ""}`}
                  style={{ left: pct(s.start), width: `max(3px, calc(${pct(s.end)} - ${pct(s.start)}))`, background: speakerColor(spk.color) }}
                  onClick={(e) => {
                    e.stopPropagation();
                    props.onSelect(s);
                  }}
                  aria-label={`${spk.display_name} ${fmtTime(s.start)}`}
                />
              ))}
          </div>
        ))}
        <div className="tl-playhead" style={{ left: pct(time) }} />
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------- segment row

const SegmentRow = forwardRef<HTMLDivElement, {
  jobId: string;
  seg: DraftSegment;
  speakers: DraftSpeaker[];
  speaker?: DraftSpeaker;
  active: boolean;
  dirty: boolean;
  readOnly: boolean;
  duration: number;
  onFocus: () => void;
  onChange: (c: Omit<SegmentPatch, "segment_id">) => void;
}>(function SegmentRow({ jobId, seg, speakers, speaker, active, dirty, readOnly, duration, onFocus, onChange }, ref) {
  const [timing, setTiming] = useState(false);
  const [showOriginal, setShowOriginal] = useState(false);
  return (
    <div
      ref={ref}
      className={`seg ${active ? "active" : ""} ${dirty ? "dirty" : ""}`}
      style={{ borderLeftColor: speaker ? speakerColor(speaker.color) : undefined }}
      data-testid={`seg-${seg.segment_id}`}
    >
      <div className="seg-meta">
        <button className="seg-time mono" onClick={onFocus} title="Seek video to this line">
          {fmtTime(seg.start, true)} – {fmtTime(seg.end, true)}
        </button>
        <PlayButton src={urls.originalSegment(jobId, seg.segment_id)} label="Play original line" />
        <select
          className="input input-sm"
          value={seg.speaker_id}
          disabled={readOnly}
          onChange={(e) => onChange({ speaker_id: e.target.value })}
          aria-label="Speaker"
        >
          {speakers.map((s) => (
            <option key={s.speaker_id} value={s.speaker_id}>
              {s.display_name}
            </option>
          ))}
        </select>
        <span className="muted small" title="Voice source of this speaker">
          {seg.voice_source ? VOICE_SOURCE_LABEL[seg.voice_source].split(" ")[0] : ""}
        </span>
        <select
          className={`input input-sm status-${seg.review_status}`}
          value={seg.review_status}
          disabled={readOnly}
          onChange={(e) => onChange({ review_status: e.target.value as SegmentReviewStatus })}
          aria-label="Review status"
        >
          <option value="pending">Pending</option>
          <option value="edited">Edited</option>
          <option value="approved">Approved</option>
        </select>
        {seg.confidence !== null && (
          <span className={`small ${seg.confidence < 0.6 ? "warn-text" : "muted"}`} title="ASR confidence">
            {Math.round(seg.confidence * 100)}%
          </span>
        )}
        {!readOnly && (
          <button className="link-btn small" onClick={() => setTiming((t) => !t)}>
            {timing ? "Done" : "Timing"}
          </button>
        )}
      </div>
      {timing && (
        <div className="row gap-sm seg-timing">
          <label className="small">
            Start{" "}
            <input className="input input-sm mono" type="number" step={0.05} min={0} max={duration} value={seg.start}
              onChange={(e) => onChange({ start: Number(e.target.value) })} />
          </label>
          <label className="small">
            End{" "}
            <input className="input input-sm mono" type="number" step={0.05} min={0} max={duration} value={seg.end}
              onChange={(e) => onChange({ end: Number(e.target.value) })} />
          </label>
          <span className="muted small">{(seg.end - seg.start).toFixed(2)} s window</span>
        </div>
      )}
      <div className="seg-original" onClick={() => !readOnly && setShowOriginal(true)}>
        {showOriginal && !readOnly ? (
          <textarea className="input" rows={2} value={seg.original_text} aria-label="Original text"
            onChange={(e) => onChange({ original_text: e.target.value })} onBlur={() => setShowOriginal(false)} autoFocus />
        ) : (
          <span title={readOnly ? undefined : "Click to correct the transcript"}>{seg.original_text}</span>
        )}
      </div>
      <textarea
        className="input hindi"
        lang="hi"
        rows={Math.min(5, Math.max(2, Math.ceil(seg.translated_text.length / 60)))}
        value={seg.translated_text}
        readOnly={readOnly}
        onFocus={onFocus}
        onChange={(e) => onChange({ translated_text: e.target.value })}
        aria-label="Hindi translation"
      />
      {seg.translated_text !== seg.machine_translation && (
        <div className="muted small">
          Machine: <span lang="hi" className="hindi-inline">{seg.machine_translation}</span>{" "}
          {!readOnly && (
            <button className="link-btn small" onClick={() => onChange({ translated_text: seg.machine_translation })}>
              Revert
            </button>
          )}
        </div>
      )}
      {seg.warnings.length > 0 && (
        <ul className="seg-warnings">
          {seg.warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
    </div>
  );
});

// ------------------------------------------------------------------------------- speaker rail
function SpeakerRail(props: {
  jobId: string;
  projectId: string;
  speakers: DraftSpeaker[];
  library: LibrarySpeaker[];
  castProfiles: VoiceProfile[];
  characters: SeriesCharacter[];
  cloneEngine: string;
  minRef: number;
  readOnly: boolean;
  saving: boolean;
  filter: string;
  onFilter: (id: string) => void;
  onEdit: (id: string, c: Partial<SpeakerPatch>) => void;
  onMerge: (source: string, target: string) => Promise<unknown>;
  onAdd: (name: string) => Promise<unknown>;
}) {
  const [newName, setNewName] = useState("");
  const usableLibrary = props.library.filter((l) => l.reference_available && !!l.reference_transcript?.trim());
  return (
    <div className="rail" aria-label="Speakers">
      <div className="section-head">
        <h2>Speakers & voices</h2>
        {props.library.length === 0 && (
          <Link className="small" to={`/projects/${props.projectId}`}>
            Project library is empty
          </Link>
        )}
      </div>
      {props.speakers.map((s) => (
        <SpeakerCard key={s.speaker_id} {...props} speaker={s} usableLibrary={usableLibrary} />
      ))}
      {!props.readOnly && (
        <form
          className="row gap-sm"
          onSubmit={async (e) => {
            e.preventDefault();
            if (!newName.trim()) return;
            await props.onAdd(newName.trim());
            setNewName("");
          }}
        >
          <input className="input input-sm" placeholder="New speaker name (to split lines)" value={newName} maxLength={80}
            onChange={(e) => setNewName(e.target.value)} aria-label="New speaker name" />
          <Button size="sm" type="submit" disabled={!newName.trim() || props.saving}>
            Add speaker
          </Button>
        </form>
      )}
    </div>
  );
}

function SpeakerCard(props: {
  jobId: string;
  projectId: string;
  speaker: DraftSpeaker;
  speakers: DraftSpeaker[];
  usableLibrary: LibrarySpeaker[];
  castProfiles: VoiceProfile[];
  characters: SeriesCharacter[];
  cloneEngine: string;
  minRef: number;
  readOnly: boolean;
  saving: boolean;
  filter: string;
  onFilter: (id: string) => void;
  onEdit: (id: string, c: Partial<SpeakerPatch>) => void;
  onMerge: (source: string, target: string) => Promise<unknown>;
}) {
  const { speaker: s, readOnly } = props;
  const [mergeTarget, setMergeTarget] = useState("");
  const character = props.characters.find((item) => item.character_id === s.character_id);
  const characterProfile = character?.voice_profiles.find((item) => item.voice_profile_id === character.active_voice_profile_id);
  const characterVoice = props.castProfiles.find((item) => item.voice_id === characterProfile?.voice_asset_id);
  const castVoice = props.castProfiles.find((item) => item.voice_id === s.voice_id);
  const short = s.usable_reference_duration < props.minRef;
  const preview =
    s.voice_source === "cast" && castVoice
      ? urls.voicePreview(castVoice.voice_id, castVoice.preview?.run_id)
      : s.voice_source === "character" && characterVoice
        ? urls.voicePreview(characterVoice.voice_id, characterVoice.preview?.run_id)
        :
    s.voice_source === "library" && s.library_speaker_id
      ? urls.libraryReference(props.projectId, s.library_speaker_id)
      :       s.voice_source === "default" && s.default_voice_id
          ? urls.defaultVoice(s.default_voice_id)
        : s.reference_available
          ? urls.jobSpeakerReference(props.jobId, s.speaker_id)
          : null;
  const setSource = (src: VoiceSource) => {
    const c: Partial<SpeakerPatch> = { voice_source: src };
    if (src === "cast") {
      const selected = props.castProfiles.find((voice) => voice.voice_id === s.voice_id) ?? props.castProfiles[0];
      c.voice_id = selected?.voice_id ?? null;
      c.engine = selected?.engine;
    }
    if (src === "character" && !s.character_id) {
      c.character_id = props.characters.find((item) => item.has_approved_voice)?.character_id ?? null;
    }
    if (src === "library" && !s.library_speaker_id && props.usableLibrary.length === 1) c.library_speaker_id = props.usableLibrary[0].speaker_id;
    if (src === "clone" || src === "library") c.engine = props.cloneEngine;
    props.onEdit(s.speaker_id, c);
  };
  const invalid =
    (s.voice_source === "cast" && !props.castProfiles.some((voice) => voice.voice_id === s.voice_id)) ||
    (s.voice_source === "character" && !props.characters.some((item) => item.character_id === s.character_id && item.has_approved_voice)) ||
    (s.voice_source === "library" && !props.usableLibrary.some((l) => l.speaker_id === s.library_speaker_id)) ||
    s.voice_source === "default" ||
    (s.voice_source === "clone" && !s.reference_available);

  return (
    <div className={`spk ${props.filter === s.speaker_id ? "focused" : ""}`} style={{ borderLeftColor: speakerColor(s.color) }}>
      <div className="row gap-sm">
        <span className="dot" style={{ background: speakerColor(s.color) }} />
        <input
          className="input input-sm spk-name"
          value={s.display_name}
          disabled={readOnly}
          maxLength={80}
          onChange={(e) => props.onEdit(s.speaker_id, { display_name: e.target.value })}
          aria-label={`Name of ${s.speaker_id}`}
        />
        <button className="link-btn small" onClick={() => props.onFilter(props.filter === s.speaker_id ? "" : s.speaker_id)}>
          {props.filter === s.speaker_id ? "Show all" : "Only this"}
        </button>
      </div>
      <div className="muted small spk-stats">
        {s.segment_count} lines · {s.total_speech_duration.toFixed(1)} s speech ·{" "}
        <span className={short ? "warn-text" : ""}>{s.usable_reference_duration.toFixed(1)} s clean reference</span>
        <span className="mono"> · {s.origin_labels.join("+") || s.speaker_id}</span>
      </div>
      {(s.suggested_character_name || props.characters.length > 0) && (
        <div className="stack-sm identity-review">
          {s.suggested_character_name && s.mapping_status !== "rejected" && (
            <div className="banner banner-info">
              Identity suggestion: <strong>{s.suggested_character_name}</strong>
              {s.identity_confidence != null && ` · ${Math.round(s.identity_confidence * 100)}%`}
              {s.identity_reason && <span> · {s.identity_reason}</span>}
              {!readOnly && (
                <span className="row gap-sm">
                  <Button size="sm" onClick={() => props.onEdit(s.speaker_id, {
                    character_id: s.suggested_character_id ?? null,
                    reject_suggestion: false,
                  })}>Confirm identity</Button>
                  <Button size="sm" variant="ghost" onClick={() => props.onEdit(s.speaker_id, { reject_suggestion: true })}>
                    Reject
                  </Button>
                </span>
              )}
            </div>
          )}
          {props.characters.length > 0 && (
            <label className="field">
              <span>Confirmed series identity (independent of voice choice)</span>
              <select className="input input-sm" value={s.character_id ?? ""} disabled={readOnly}
                onChange={(e) => props.onEdit(s.speaker_id, { character_id: e.target.value || null, reject_suggestion: false })}
                aria-label={`Character identity for ${s.display_name}`}>
                <option value="">Unassigned</option>
                {props.characters.map((item) => (
                  <option key={item.character_id} value={item.character_id}>{item.canonical_name}</option>
                ))}
              </select>
            </label>
          )}
        </div>
      )}
      <div className="voice-src" role="radiogroup" aria-label={`Voice source for ${s.display_name}`}>
        {(["cast", "character", "clone", "library"] as VoiceSource[]).map((src) => (
          <label key={src} className={`radio ${s.voice_source === src ? "on" : ""}`}>
            <input
              type="radio"
              name={`src-${s.speaker_id}`}
              checked={s.voice_source === src}
              disabled={readOnly || (src === "clone" && !s.reference_available) ||
                (src === "cast" && props.castProfiles.length === 0) ||
                (src === "library" && props.usableLibrary.length === 0) ||
                (src === "character" && !props.characters.some((item) => item.has_approved_voice))}
              onChange={() => setSource(src)}
            />
            {VOICE_SOURCE_LABEL[src]}
          </label>
        ))}
      </div>
      {s.voice_source === "cast" && (
        <select
          className="input input-sm"
          value={s.voice_id ?? ""}
          disabled={readOnly}
          onChange={(e) => {
            const voice = props.castProfiles.find((item) => item.voice_id === e.target.value);
            props.onEdit(s.speaker_id, { voice_id: e.target.value || null, engine: voice?.engine });
          }}
          aria-label="Approved voice profile"
        >
          <option value="">{props.castProfiles.length ? "Select an approved voice profile…" : "No approved Parler/IndicF5 profiles"}</option>
          {props.castProfiles.map((voice) => (
            <option key={voice.voice_id} value={voice.voice_id}>{voice.voice_name} · {voice.engine}</option>
          ))}
        </select>
      )}
      {s.voice_source === "character" && (
        <select
          className="input input-sm"
          value={s.character_id ?? ""}
          disabled={readOnly}
          onChange={(e) => props.onEdit(s.speaker_id, { character_id: e.target.value || null })}
          aria-label="Character voice"
        >
          <option value="">Select a character with an approved voice…</option>
          {props.characters.filter((item) => item.has_approved_voice).map((item) => (
            <option key={item.character_id} value={item.character_id}>{item.canonical_name}</option>
          ))}
        </select>
      )}
      {s.voice_source === "library" && (
        <select
          className="input input-sm"
          value={s.library_speaker_id ?? ""}
          disabled={readOnly}
          onChange={(e) => props.onEdit(s.speaker_id, { library_speaker_id: e.target.value || null })}
          aria-label="Library speaker"
        >
          <option value="">{props.usableLibrary.length ? "Select a character…" : "No library speakers with reference and transcript"}</option>
          {props.usableLibrary.map((l) => (
            <option key={l.speaker_id} value={l.speaker_id}>
              {l.display_name} ({l.reference_speech_seconds.toFixed(0)} s)
            </option>
          ))}
        </select>
      )}
      {s.voice_source === "default" && <Badge tone="warn">Legacy voice source; choose a current engine profile explicitly.</Badge>}
      <div className="row gap-sm">
        {preview ? <PlayButton src={preview} label="Preview reference voice" /> : <span className="muted small">No preview</span>}
        <span className="small muted">Preview reference</span>
        {invalid && <Badge tone="error">Select a valid voice</Badge>}
      </div>
      {s.warnings.filter((w) => s.voice_source === "clone" || !w.startsWith("Only ")).map((w, i) => (
        <div key={i} className="warn-text small">
          {w}
        </div>
      ))}
      {!readOnly && props.speakers.length > 1 && (
        <div className="row gap-sm">
          <select className="input input-sm" value={mergeTarget} onChange={(e) => setMergeTarget(e.target.value)} aria-label="Merge into">
            <option value="">Merge into…</option>
            {props.speakers
              .filter((o) => o.speaker_id !== s.speaker_id)
              .map((o) => (
                <option key={o.speaker_id} value={o.speaker_id}>
                  {o.display_name}
                </option>
              ))}
          </select>
          <Button
            size="sm"
            variant="ghost"
            disabled={!mergeTarget || props.saving}
            onClick={async () => {
              await props.onMerge(s.speaker_id, mergeTarget);
              setMergeTarget("");
            }}
          >
            Merge
          </Button>
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------------------- confirm
function ConfirmDialog({ jobId, version, library, castProfiles, characters, onClose, onConfirmed }: {
  jobId: string;
  version: number;
  library: LibrarySpeaker[];
  castProfiles: VoiceProfile[];
  characters: SeriesCharacter[];
  onClose: () => void;
  onConfirmed: () => void;
}) {
  const preview = useAsync<ConfirmPreview>(() => draftApi.preview(jobId), [jobId, version]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const name = (speaker: ConfirmPreview["speakers"][number]) =>
    speaker.voice_source === "cast"
      ? `${castProfiles.find((voice) => voice.voice_id === speaker.voice_id)?.voice_name ?? "Voice profile"} · ${speaker.engine ?? ""}`
      : speaker.voice_source === "character"
        ? `Character · ${characters.find((item) => item.character_id === speaker.character_id)?.canonical_name ?? speaker.character_name ?? "?"}`
        : speaker.voice_source === "library"
          ? `IndicF5 project speaker · ${library.find((l) => l.speaker_id === speaker.library_speaker_id)?.display_name ?? "?"}`
          : speaker.voice_source === "default"
            ? `Legacy default assignment · ${speaker.default_voice_id ?? "unavailable"}`
            : "IndicF5 from this episode";
  const p = preview.data;
  return (
    <Modal
      title="Confirm & generate the Hindi dub"
      onClose={onClose}
      wide
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Back to review
          </Button>
          <Button
            variant="primary"
            busy={busy}
            disabled={!p || p.errors.length > 0 || p.version !== version}
            onClick={async () => {
              setBusy(true);
              setError(null);
              try {
                await draftApi.confirm(jobId, version);
                onConfirmed();
              } catch (e) {
                setError(errorText(e));
                setBusy(false);
              }
            }}
          >
            Lock draft v{version} & generate
          </Button>
        </>
      }
    >
      {preview.loading ? (
        <Spinner label="Validating draft" />
      ) : (
        <div className="stack">
          <ErrorBanner error={preview.error} />
          <p>
            Draft v{version} will be locked as an immutable approved snapshot. Voices are synthesized only after this
            step; edits are no longer possible for this job.
          </p>
          {p && (
            <table className="table compact">
              <thead>
                <tr>
                  <th>Speaker</th>
                  <th className="num">Lines</th>
                  <th>Voice</th>
                </tr>
              </thead>
              <tbody>
                {p.speakers.map((s) => (
                  <tr key={s.speaker_id}>
                    <td>{s.display_name}</td>
                    <td className="num">{s.segment_count}</td>
                    <td>{name(s)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {p && p.errors.length > 0 && (
            <div className="banner banner-error">
              <strong>Fix before generating</strong>
              <ul>
                {p.errors.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
            </div>
          )}
          {p && <WarningList warnings={p.warnings} />}
          {error && <div className="banner banner-error">{error}</div>}
        </div>
      )}
    </Modal>
  );
}
