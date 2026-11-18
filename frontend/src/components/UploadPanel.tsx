import { useRef, useState } from "react";
import { videoApi } from "../api";
import { errorText } from "../api/client";
import type { Meta, MixMode, Project, VideoUploadResponse } from "../api/types";
import { Button, Field, fmtBytes, fmtTime } from "./ui";

const MIX_LABEL: Record<MixMode, string> = {
  preserve_background: "Preserve background (music + SFX, original dialogue removed)",
  dub_only: "Dub only (Hindi dialogue, no original audio)",
  original_plus_dub: "Original ducked under the dub",
};

interface Props {
  project: Project;
  meta?: Meta;
  onUploaded: (r: VideoUploadResponse) => void;
}

/** Drag-and-drop video upload with real byte progress; analysis starts on the server after upload. */
export default function UploadPanel({ project, meta, onUploaded }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [title, setTitle] = useState("");
  const [source, setSource] = useState(project.settings.source_language ?? "");
  const [mix, setMix] = useState<MixMode>(project.settings.mix_mode);
  const [minSpk, setMinSpk] = useState("");
  const [maxSpk, setMaxSpk] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const abortRef = useRef<(() => void) | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const exts = meta?.video_extensions ?? [".mp4", ".mkv", ".mov", ".m4v", ".webm", ".avi"];
  const maxMb = meta?.max_upload_mb ?? 8192;

  const choose = (f: File | undefined) => {
    setError(null);
    setDuration(null);
    if (!f) return;
    const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
    if (!exts.includes(ext)) {
      setError(`Unsupported file type "${ext}". Allowed: ${exts.join(", ")}`);
      return;
    }
    if (f.size > maxMb * 1024 * 1024) {
      setError(`File is ${fmtBytes(f.size)}; the server limit is ${maxMb} MB.`);
      return;
    }
    setFile(f);
    setTitle(f.name.replace(/\.[^.]+$/, ""));
    // Local metadata only (the server re-probes with ffprobe after upload).
    const url = URL.createObjectURL(f);
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => {
      if (Number.isFinite(v.duration)) setDuration(v.duration);
      URL.revokeObjectURL(url);
    };
    v.onerror = () => URL.revokeObjectURL(url);
    v.src = url;
  };

  const start = async () => {
    if (!file) return;
    setError(null);
    setProgress(0);
    const handle = videoApi.upload(
      project.project_id,
      file,
      {
        title: title.trim() || undefined,
        source_language: source || undefined,
        mix_mode: mix,
        min_speakers: minSpk ? Number(minSpk) : undefined,
        max_speakers: maxSpk ? Number(maxSpk) : undefined,
      },
      setProgress,
    );
    abortRef.current = handle.abort;
    try {
      const r = await handle.promise;
      setFile(null);
      setProgress(null);
      onUploaded(r);
    } catch (e) {
      setError(errorText(e));
      setProgress(null);
    } finally {
      abortRef.current = null;
    }
  };

  const uploading = progress !== null;
  return (
    <div className="panel upload">
      <div
        className={`dropzone ${drag ? "is-drag" : ""} ${file ? "has-file" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          if (!uploading) choose(e.dataTransfer.files[0]);
        }}
        onClick={() => !uploading && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        aria-label="Choose a video file"
      >
        <input
          ref={inputRef}
          type="file"
          hidden
          accept={exts.join(",")}
          data-testid="video-input"
          onChange={(e) => choose(e.target.files?.[0])}
        />
        {file ? (
          <div className="file-info">
            <strong>{file.name}</strong>
            <span className="muted">
              {fmtBytes(file.size)} · {duration !== null ? fmtTime(duration) : "reading duration…"}
            </span>
          </div>
        ) : (
          <div>
            <strong>Drop a video here</strong> or click to browse
            <div className="muted small">
              {exts.join(" ")} · up to {maxMb >= 1024 ? `${(maxMb / 1024).toFixed(0)} GB` : `${maxMb} MB`}
            </div>
          </div>
        )}
      </div>

      <div className="grid-3">
        <Field label="Episode title">
          <input className="input" value={title} maxLength={200} onChange={(e) => setTitle(e.target.value)} disabled={uploading} />
        </Field>
        <Field label="Source language" hint="Set it explicitly for videos that open with music.">
          <select className="input" value={source} onChange={(e) => setSource(e.target.value)} disabled={uploading}>
            <option value="">Auto-detect</option>
            {Object.entries(meta?.source_languages ?? { en: "English", zh: "Chinese (Mandarin)", ko: "Korean" }).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Target language">
          <select className="input" value="hi" disabled>
            <option value="hi">Hindi</option>
          </select>
        </Field>
        <Field label="Audio mix">
          <select className="input" value={mix} onChange={(e) => setMix(e.target.value as MixMode)} disabled={uploading}>
            {(Object.keys(MIX_LABEL) as MixMode[]).map((m) => (
              <option key={m} value={m}>
                {MIX_LABEL[m]}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Min speakers" hint="Optional; helps diarization.">
          <input className="input" type="number" min={1} max={20} value={minSpk} onChange={(e) => setMinSpk(e.target.value)} disabled={uploading} />
        </Field>
        <Field label="Max speakers">
          <input className="input" type="number" min={1} max={20} value={maxSpk} onChange={(e) => setMaxSpk(e.target.value)} disabled={uploading} />
        </Field>
      </div>

      {error && <div className="banner banner-error" role="alert">{error}</div>}
      {uploading && (
        <div className="upload-progress" aria-label="Upload progress">
          <div className="bar">
            <div className="bar-fill" style={{ width: `${Math.round((progress ?? 0) * 100)}%` }} />
          </div>
          <span className="muted small">
            {progress! < 1 ? `Uploading ${Math.round(progress! * 100)}%` : "Upload complete — validating media…"}
          </span>
        </div>
      )}
      <div className="row end gap">
        {uploading ? (
          <Button variant="ghost" onClick={() => abortRef.current?.()}>
            Cancel upload
          </Button>
        ) : (
          file && (
            <Button variant="ghost" onClick={() => setFile(null)}>
              Clear
            </Button>
          )
        )}
        <Button variant="primary" onClick={start} disabled={!file || uploading}>
          Upload & analyze
        </Button>
      </div>
    </div>
  );
}
