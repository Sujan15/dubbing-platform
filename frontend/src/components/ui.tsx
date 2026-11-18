import { useEffect, useRef, useState, type ReactNode } from "react";
import { errorText } from "../api/client";
import type { JobState } from "../api/types";

// ---------------------------------------------------------------------------- formatting
export function fmtTime(seconds: number | null | undefined, withMs = false): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "--:--";
  const s = Math.max(0, seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const ss = withMs ? sec.toFixed(2).padStart(5, "0") : Math.floor(sec).toString().padStart(2, "0");
  return h ? `${h}:${m.toString().padStart(2, "0")}:${ss}` : `${m.toString().padStart(2, "0")}:${ss}`;
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function fmtDuration(fromIso: string | null | undefined, toIso?: string | null): string {
  if (!fromIso) return "—";
  const ms = (toIso ? new Date(toIso).getTime() : Date.now()) - new Date(fromIso).getTime();
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

export const SPEAKER_COLORS = [
  "#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa",
  "#f87171", "#22d3ee", "#a3e635", "#fb923c", "#e879f9",
];
export const speakerColor = (i: number) => SPEAKER_COLORS[((i % SPEAKER_COLORS.length) + SPEAKER_COLORS.length) % SPEAKER_COLORS.length];

export const STATE_LABEL: Record<JobState, string> = {
  queued: "Queued",
  running: "Analyzing",
  awaiting_review: "Awaiting review",
  generating: "Generating",
  succeeded: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

export const VOICE_SOURCE_LABEL = {
  cast: "Approved Voice Library profile",
  character: "Series character voice",
  clone: "IndicF5 from this video",
  library: "IndicF5 project speaker",
  default: "Legacy default voice",
} as const;

export const STAGE_LABEL: Record<string, string> = {
  validate: "Validate input",
  extract: "Extract audio",
  separate: "Separate vocals / background",
  transcribe: "Transcribe (ASR)",
  diarize: "Speaker diarization",
  align: "Align speakers to words",
  translate: "Hindi translation",
  voices: "Speaker reference audio",
  prepare: "Load cached analysis",
  resolve_voices: "Voice resolution",
  synthesize: "TTS + timing adjustment",
  mix: "Audio mixing",
  mux: "Video muxing",
  validate_output: "Output validation",
};

// ---------------------------------------------------------------------------- primitives
type BtnProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md";
  busy?: boolean;
};

export function Button({ variant = "secondary", size = "md", busy, children, className = "", disabled, ...rest }: BtnProps) {
  return (
    <button
      className={`btn btn-${variant} btn-${size} ${className}`}
      disabled={disabled || busy}
      aria-busy={busy || undefined}
      {...rest}
    >
      {busy && <span className="spinner spinner-inline" aria-hidden />}
      {children}
    </button>
  );
}

export function StatusBadge({ state }: { state: JobState }) {
  return <span className={`badge badge-${state}`}>{STATE_LABEL[state]}</span>;
}

export function Badge({ tone = "neutral", children, title }: { tone?: "neutral" | "warn" | "error" | "ok" | "info"; children: ReactNode; title?: string }) {
  return (
    <span className={`badge badge-tone-${tone}`} title={title}>
      {children}
    </span>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div className="loading" role="status">
      <span className="spinner" aria-hidden /> {label}…
    </div>
  );
}

export function EmptyState({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function ErrorBanner({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  if (!error) return null;
  return (
    <div className="banner banner-error" role="alert">
      <span>{errorText(error)}</span>
      {onRetry && (
        <Button size="sm" variant="ghost" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

export function WarningList({ warnings, title = "Warnings" }: { warnings: string[]; title?: string }) {
  if (!warnings.length) return null;
  return (
    <div className="banner banner-warn" role="status">
      <strong>{title}</strong>
      <ul>
        {warnings.map((w, i) => (
          <li key={i}>{w}</li>
        ))}
      </ul>
    </div>
  );
}

export function Modal({ title, onClose, children, footer, wide }: { title: string; onClose: () => void; children: ReactNode; footer?: ReactNode; wide?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    ref.current?.querySelector<HTMLElement>("input,textarea,select,button")?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`modal ${wide ? "modal-wide" : ""}`} role="dialog" aria-modal="true" aria-label={title} ref={ref}>
        <header className="modal-head">
          <h2>{title}</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        <div className="modal-body">{children}</div>
        {footer && <footer className="modal-foot">{footer}</footer>}
      </div>
    </div>
  );
}

/** Single shared <audio> so starting one preview stops the previous one. */
let sharedAudio: HTMLAudioElement | null = null;
let sharedOwner: ((playing: boolean) => void) | null = null;

export function PlayButton({ src, label = "Play", disabled }: { src: string; label?: string; disabled?: boolean }) {
  const [playing, setPlaying] = useState(false);
  const [failed, setFailed] = useState(false);
  useEffect(() => () => {
    if (sharedOwner === setPlaying) sharedAudio?.pause();
  }, []);
  const toggle = () => {
    if (!sharedAudio) sharedAudio = new Audio();
    const a = sharedAudio;
    if (playing) {
      a.pause();
      setPlaying(false);
      return;
    }
    sharedOwner?.(false);
    sharedOwner = setPlaying;
    a.src = src;
    a.onended = () => setPlaying(false);
    a.onerror = () => {
      setPlaying(false);
      setFailed(true);
    };
    setFailed(false);
    a.play().then(() => setPlaying(true)).catch(() => setFailed(true));
  };
  return (
    <button
      type="button"
      className={`play-btn ${playing ? "is-playing" : ""} ${failed ? "is-failed" : ""}`}
      onClick={toggle}
      disabled={disabled}
      aria-label={playing ? `Stop ${label}` : label}
      title={failed ? "Audio unavailable" : label}
    >
      {playing ? "■" : "▶"}
    </button>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: ReactNode }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}
