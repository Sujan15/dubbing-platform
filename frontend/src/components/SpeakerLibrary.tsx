import { useRef, useState } from "react";
import { speakerApi, urls, voiceApi } from "../api";
import { errorText } from "../api/client";
import type { LibrarySpeaker, Meta } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { Badge, Button, EmptyState, ErrorBanner, Field, Modal, PlayButton, Spinner, fmtDate } from "./ui";

const SOURCE_LABEL = { cloned: "Cloned", uploaded: "Uploaded", default_promoted: "From default voice" } as const;

interface Props {
  projectId: string;
  speakers: LibrarySpeaker[];
  meta?: Meta;
  onChanged: () => void;
}

export default function SpeakerLibrary({ projectId, speakers, meta, onChanged }: Props) {
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState<LibrarySpeaker | null>(null);
  const [deleting, setDeleting] = useState<LibrarySpeaker | null>(null);
  const [error, setError] = useState<unknown>(null);

  return (
    <div className="stack-lg">
      <div className="section-head">
        <div>
          <h2>Project Speaker Library</h2>
          <p className="muted small">
            Characters and their reference recordings. Map a detected speaker to a library character in any episode to
            keep the same voice across the series.
          </p>
        </div>
        <Button variant="primary" onClick={() => setAdding(true)}>
          Add speaker
        </Button>
      </div>
      <ErrorBanner error={error} />
      {speakers.length === 0 ? (
        <EmptyState title="No library speakers yet">
          Add a character and upload 15–20 s of clean speech, or save a voice from a reviewed or completed episode.
        </EmptyState>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>Speaker</th>
              <th>Reference</th>
              <th className="num">Speech</th>
              <th>Source</th>
              <th>Notes</th>
              <th>Created</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {speakers.map((s) => (
              <tr key={s.speaker_id}>
                <td className="strong">{s.display_name}</td>
                <td>
                  {s.reference_available ? (
                    <div className="row gap-sm">
                      <PlayButton src={urls.libraryReference(projectId, s.speaker_id, s.reference_updated_at)} label={`Play ${s.display_name}`} />
                      <span className="muted small">{s.reference_duration_seconds.toFixed(1)} s</span>
                    </div>
                  ) : (
                    <Badge tone="warn">No reference</Badge>
                  )}
                  {s.reference_available && !s.reference_transcript && (
                    <div className="warn-text small">Add a reference transcript before using this IndicF5 voice.</div>
                  )}
                  {s.warnings.map((w, i) => (
                    <div key={i} className="warn-text small">
                      {w}
                    </div>
                  ))}
                </td>
                <td className="num">{s.reference_available ? `${s.reference_speech_seconds.toFixed(1)} s` : "—"}</td>
                <td>{s.source ? SOURCE_LABEL[s.source] : "—"}</td>
                <td className="muted small">
                  <span className="clamp">{s.notes || "—"}</span>
                </td>
                <td className="muted small">{fmtDate(s.created_at)}</td>
                <td>
                  <div className="row gap-sm end">
                    <Button size="sm" variant="ghost" onClick={() => setEditing(s)}>
                      Edit
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setDeleting(s)}>
                      Delete
                    </Button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <DefaultVoiceBank />

      {(adding || editing) && (
        <SpeakerForm
          projectId={projectId}
          speaker={editing}
          meta={meta}
          onClose={() => {
            setAdding(false);
            setEditing(null);
          }}
          onSaved={() => {
            setAdding(false);
            setEditing(null);
            onChanged();
          }}
        />
      )}
      {deleting && (
        <Modal
          title={`Delete ${deleting.display_name}?`}
          onClose={() => setDeleting(null)}
          footer={
            <>
              <Button variant="ghost" onClick={() => setDeleting(null)}>
                Cancel
              </Button>
              <Button
                variant="danger"
                onClick={async () => {
                  try {
                    await speakerApi.remove(projectId, deleting.speaker_id);
                    onChanged();
                  } catch (e) {
                    setError(e);
                  }
                  setDeleting(null);
                }}
              >
                Delete speaker
              </Button>
            </>
          }
        >
          <p>
            The reference recording is removed permanently. Episodes already generated keep their audio; open reviews
            that select this speaker block deletion.
          </p>
        </Modal>
      )}
    </div>
  );
}

function SpeakerForm({ projectId, speaker, meta, onClose, onSaved }: {
  projectId: string;
  speaker: LibrarySpeaker | null;
  meta?: Meta;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [name, setName] = useState(speaker?.display_name ?? "");
  const [notes, setNotes] = useState(speaker?.notes ?? "");
  const [transcript, setTranscript] = useState(speaker?.reference_transcript ?? "");
  const [file, setFile] = useState<File | null>(null);
  const [replace, setReplace] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const exts = meta?.reference_extensions ?? [".wav", ".flac", ".mp3", ".m4a", ".ogg"];
  const needsReplace = !!speaker?.reference_available && !!file;

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      let s = speaker;
      if (!s) s = await speakerApi.create(projectId, { display_name: name.trim(), notes });
      if (file) {
        if (!transcript.trim()) throw new Error("Enter the words spoken in the reference recording for IndicF5.");
        setProgress(0);
        await speakerApi.uploadReference(projectId, s.speaker_id, file, replace, transcript.trim(), setProgress).promise;
      }
      if (speaker && (name.trim() !== s.display_name || notes !== s.notes ||
        (!file && transcript !== (s.reference_transcript ?? "")))) {
        s = await speakerApi.update(projectId, s.speaker_id, {
          display_name: name.trim(), notes,
          ...(!file ? { reference_transcript: transcript } : {}),
        });
      }
      onSaved();
    } catch (err) {
      setError(errorText(err));
      setBusy(false);
      setProgress(null);
    }
  };

  return (
    <Modal title={speaker ? `Edit ${speaker.display_name}` : "Add library speaker"} onClose={onClose}>
      <form className="stack" onSubmit={save}>
        <Field label="Character name">
          <input className="input" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} required />
        </Field>
        <Field label="Notes">
          <textarea className="input" rows={2} value={notes} maxLength={2000} onChange={(e) => setNotes(e.target.value)} />
        </Field>
        <Field label="Reference transcript (IndicF5)" hint="Enter the exact words spoken in the reference audio.">
          <textarea className="input" rows={3} value={transcript} maxLength={4000}
            onChange={(e) => setTranscript(e.target.value)} />
        </Field>
        <Field
          label={speaker?.reference_available ? "Replace reference recording" : "Reference recording"}
          hint="15–20 s of one person speaking, no music or other voices. Converted to mono 16-bit WAV automatically; transcript is required."
        >
          <input
            ref={fileRef}
            className="input"
            type="file"
            accept={exts.join(",")}
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </Field>
        {needsReplace && (
          <label className="check">
            <input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} />
            Replace the existing reference (the old file is kept in the speaker's history folder)
          </label>
        )}
        {progress !== null && <div className="muted small">Uploading {Math.round(progress * 100)}% — validating audio…</div>}
        {error && <div className="banner banner-error">{error}</div>}
        <div className="row end gap">
          <Button type="button" variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" variant="primary" busy={busy} disabled={!name.trim() || (needsReplace && !replace)}>
            Save
          </Button>
        </div>
      </form>
    </Modal>
  );
}

export function DefaultVoiceBank() {
  const voices = useAsync((s) => voiceApi.defaults(s), []);
  return (
    <div className="stack">
      <div className="section-head">
        <div>
          <h2>Default Hindi Voice Bank</h2>
          <p className="muted small">
            Historical audio kept for preview only. These voices are not available for synthesis.
          </p>
        </div>
      </div>
      <ErrorBanner error={voices.error} onRetry={voices.reload} />
      {voices.loading && !voices.data ? (
        <Spinner label="Loading default voices" />
      ) : !voices.data?.voices.length ? (
        <EmptyState title="No default voices installed">
          No historical default voice recordings are installed.
        </EmptyState>
      ) : (
        <div className="voice-grid">
          {voices.data.voices.map((v) => (
            <div key={v.voice_id} className="voice-card">
              <PlayButton src={urls.defaultVoice(v.voice_id)} label={`Play ${v.label}`} />
              <div>
                <div className="strong">{v.label}</div>
                <div className="muted small">
                  {v.voice_id} · {v.speech_seconds.toFixed(1)} s{v.gender ? ` · ${v.gender}` : ""}
                </div>
                {v.description && <div className="small">{v.description}</div>}
              </div>
            </div>
          ))}
        </div>
      )}
      {!!voices.data?.invalid.length && (
        <div className="muted small">
          Skipped invalid files: {voices.data.invalid.map((i) => `${i.voice_id} (${i.problem})`).join(", ")}
        </div>
      )}
    </div>
  );
}
