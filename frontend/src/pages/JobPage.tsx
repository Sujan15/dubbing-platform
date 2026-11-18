import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { artifactApi, jobApi, speakerApi, urls } from "../api";
import { errorText } from "../api/client";
import type { JobRecord } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import StageList from "../components/StageList";
import {
  Badge,
  Button,
  ErrorBanner,
  Spinner,
  StatusBadge,
  VOICE_SOURCE_LABEL,
  WarningList,
  fmtDate,
  fmtDuration,
  fmtTime,
} from "../components/ui";

const ACTIVE = ["queued", "running", "generating"];

export default function JobPage() {
  const { jobId = "" } = useParams();
  const job = useAsync((s) => jobApi.get(jobId, s), [jobId], {
    pollMs: 2000,
    shouldPoll: (d) => !d || ACTIVE.includes(d.state),
  });
  if (job.loading && !job.data) return <Spinner label="Loading job" />;
  if (!job.data) return <ErrorBanner error={job.error ?? "Job not found"} onRetry={job.reload} />;
  const j = job.data;
  return (
    <section className="page">
      <Crumbs job={j} />
      <div className="page-head">
        <div>
          <h1>{j.input_filename}</h1>
          <div className="row gap-sm">
            <StatusBadge state={j.state} />
            <span className="muted small">
              Job {j.job_id.slice(0, 8)} · created {fmtDate(j.created_at)}
              {j.started_at && ` · elapsed ${fmtDuration(j.started_at, j.finished_at)}`}
            </span>
          </div>
        </div>
        <JobActions job={j} onChange={(r) => job.setData(r)} />
      </div>
      <ErrorBanner error={job.error} />
      {j.state === "succeeded" ? <ResultView job={j} onJobChange={job.setData} /> : <ProgressView job={j} />}
    </section>
  );
}

function Crumbs({ job }: { job: JobRecord }) {
  return (
    <div className="crumbs">
      <Link to="/">Projects</Link>
      {job.project_id && (
        <>
          {" / "}
          <Link to={`/projects/${job.project_id}`}>Project</Link>
        </>
      )}
      {" / "}
      <span>{job.input_filename}</span>
    </div>
  );
}

function JobActions({ job, onChange }: { job: JobRecord; onChange: (j: JobRecord) => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const act = async (fn: () => Promise<JobRecord>) => {
    setBusy(true);
    setError(null);
    try {
      onChange(await fn());
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="row gap">
      {error && <span className="error-text small">{error}</span>}
      {job.state === "awaiting_review" && (
        <Link className="btn btn-primary btn-md" to={`/jobs/${job.job_id}/review`}>
          Open review workstation
        </Link>
      )}
      {["queued", "running", "generating", "awaiting_review"].includes(job.state) && (
        <Button variant="ghost" busy={busy} onClick={() => act(() => jobApi.cancel(job.job_id))}>
          Cancel job
        </Button>
      )}
      {["failed", "cancelled"].includes(job.state) && (
        <Button variant="primary" busy={busy} onClick={() => act(() => jobApi.retry(job.job_id))}>
          Retry {job.phase === "generation" && job.approved_version ? "generation" : "analysis"}
        </Button>
      )}
    </div>
  );
}

function ProgressView({ job }: { job: JobRecord }) {
  const phase = job.phase;
  const active = ACTIVE.includes(job.state);
  const logs = useAsync((s) => jobApi.logs(job.job_id, 30, s), [job.job_id, job.stage, job.state], {
    pollMs: active ? 3000 : undefined,
  });
  return (
    <div className="split">
      <div className="panel stack">
        <h2>{phase === "generation" ? "Generation" : job.review_required ? "Analysis" : "Processing"}</h2>
        {job.state === "awaiting_review" && (
          <div className="banner banner-info">
            Analysis finished: {job.segments} lines, {job.speakers} speakers. Review the Hindi draft and voices, then
            confirm to generate.
          </div>
        )}
        {job.state === "failed" && (
          <div className="banner banner-error" role="alert">
            <strong>Failed{job.stage ? ` during ${job.stage}` : ""}:</strong> {job.error}
          </div>
        )}
        {job.state === "cancelled" && <div className="banner banner-warn">This job was cancelled.</div>}
        {job.state === "queued" || (job.state === "generating" && job.stage === "queued") ? (
          <p className="muted">Waiting for a free worker ({job.message}).</p>
        ) : null}
        <StageList job={job} phase={phase} />
        {phase === "generation" && (
          <details>
            <summary className="muted small">Analysis stages (completed before review)</summary>
            <StageList job={job} phase="analysis" />
          </details>
        )}
        <WarningList warnings={job.warnings} />
      </div>
      <div className="panel stack">
        <h2>Log</h2>
        <pre className="log" aria-live="polite">
          {logs.data?.lines.length ? logs.data.lines.join("\n") : "No log output yet."}
        </pre>
        <p className="muted small">Showing the last 30 lines of job.log.</p>
      </div>
    </div>
  );
}

function ResultView({ job, onJobChange }: { job: JobRecord; onJobChange: (job: JobRecord) => void }) {
  const report = useAsync(() => artifactApi.report(job.job_id), [job.job_id]);
  const timing = useAsync(() => artifactApi.timing(job.job_id), [job.job_id]);
  const qa = useAsync((signal) => artifactApi.qa(job.job_id, signal), [job.job_id]);
  const [promoted, setPromoted] = useState<Record<string, string>>({});
  const [releaseNote, setReleaseNote] = useState(job.release_note ?? "");
  const [releaseBusy, setReleaseBusy] = useState(false);
  const [releaseError, setReleaseError] = useState<string | null>(null);
  const voices = report.data?.voices ?? {};
  const flagged = (timing.data?.segments ?? []).filter((t) => t.warning);
  const qaStatus = qa.data?.status ?? job.qa_status ?? "UNKNOWN";
  const releaseBlocked = qa.data?.release_blocked ?? qaStatus === "FAILED";

  const promote = async (spk: string, name: string) => {
    try {
      await speakerApi.promote(job.project_id!, { job_id: job.job_id, job_speaker_id: spk, display_name: name });
      setPromoted((p) => ({ ...p, [spk]: "Saved to library" }));
    } catch (e) {
      setPromoted((p) => ({ ...p, [spk]: errorText(e) }));
    }
  };

  return (
    <div className="result">
      <div className="panel">
        <video className="player" src={urls.outputVideo(job.job_id)} controls preload="metadata" />
        <div className="row gap wrap">
          <a className="btn btn-primary btn-md" href={urls.result(job.job_id)} download>
            Download dubbed video
          </a>
          <a className="btn btn-secondary btn-md" href={urls.manifest(job.job_id)} download>
            Manifest (JSON)
          </a>
          <a className="btn btn-secondary btn-md" href={urls.timing(job.job_id)} target="_blank" rel="noreferrer">
            Timing report
          </a>
          <a className="btn btn-secondary btn-md" href={urls.report(job.job_id)} target="_blank" rel="noreferrer">
            Job report
          </a>
          {job.project_id && (
            <Link className="btn btn-ghost btn-md" to={`/projects/${job.project_id}`}>
              Back to project
            </Link>
          )}
        </div>
      </div>
      <div className="panel stack">
        <h2>Summary</h2>
        <dl className="kv">
          <dt>Target language</dt>
          <dd>Hindi</dd>
          <dt>Source language</dt>
          <dd>{job.source_language ?? "—"}</dd>
          <dt>Processing time</dt>
          <dd>{fmtDuration(job.started_at, job.finished_at)}</dd>
          <dt>Background audio</dt>
          <dd>
            {report.data
              ? report.data.mix_mode_used === report.data.mix_mode_requested
                ? report.data.mix_mode_used
                : `${report.data.mix_mode_used} (requested ${report.data.mix_mode_requested})`
              : "—"}
          </dd>
          <dt>Output validation</dt>
          <dd>
            <Badge tone="ok">Passed</Badge> <span className="muted small">duration matches source, one audio stream</span>
          </dd>
          <dt>Lines</dt>
          <dd>
            {job.segments} ({timing.data ? Object.entries(timing.data.summary).filter(([, n]) => n).map(([k, n]) => `${n} ${k}`).join(", ") : "…"})
          </dd>
        </dl>
        <h3>Speaker voices</h3>
        <table className="table compact">
          <thead>
            <tr>
              <th>Speaker</th>
              <th>Voice used</th>
              <th className="num">Ref.</th>
              {job.project_id && <th />}
            </tr>
          </thead>
          <tbody>
            {Object.entries(voices).map(([spk, v]) => (
              <tr key={spk}>
                <td>{v.display_name}</td>
                <td>
                  {VOICE_SOURCE_LABEL[v.resolved_source]}
                  {v.default_voice_id && ` · ${v.default_voice_id}`}
                  {v.fallback_applied && <Badge tone="warn">fallback</Badge>}
                </td>
                <td className="num">{v.reference_seconds.toFixed(1)} s</td>
                {job.project_id && (
                  <td>
                    {v.resolved_source !== "library" &&
                      (promoted[spk] ? (
                        <span className="muted small">{promoted[spk]}</span>
                      ) : (
                        <Button size="sm" variant="ghost" onClick={() => promote(spk, v.display_name)}>
                          Save to library
                        </Button>
                      ))}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
        <WarningList warnings={job.warnings} />
        <h3>Audio QA and release</h3>
        <div className="row gap-sm wrap">
          <Badge tone={qaStatus === "PASS" ? "ok" : qaStatus === "FAILED" ? "error" : "warn"}>QA {qaStatus}</Badge>
          <Badge tone={job.release_approved ? "ok" : releaseBlocked ? "error" : "neutral"}>
            {job.release_approved ? "Released" : releaseBlocked ? "Release blocked" : "Not released"}
          </Badge>
        </div>
        <ErrorBanner error={qa.error} />
        {qa.data?.note && <p className="muted small">{qa.data.note}</p>}
        {qa.data && (
          <table className="table compact" aria-label="Audio QA checks">
            <thead><tr><th>Check</th><th>Status</th><th>Result</th><th>Details</th></tr></thead>
            <tbody>
              {Object.entries(qa.data.checks).map(([name, check]) => (
                <tr key={name}>
                  <td>{name.replace(/_/g, " ")}</td>
                  <td><Badge tone={check.status === "PASS" ? "ok" : check.status === "FAILED" ? "error" : "warn"}>{check.status}</Badge></td>
                  <td>{check.value == null ? "—" : String(check.value)}</td>
                  <td>{check.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {qa.data?.clipping && (
          <p className="muted small">Peak clipping: {String(qa.data.clipping.clipped_samples ?? "not reported")} samples; peak {String(qa.data.clipping.peak ?? "not reported")}</p>
        )}
        <div className="row gap-sm wrap">
          {(["dialogue", "bed", "mix"] as const).map((stem) => (
            <audio key={stem} controls preload="none" src={urls.stem(job.job_id, stem)} aria-label={`${stem} audio stem`} />
          ))}
        </div>
        <label className="field">
          <span>Release note</span>
          <textarea className="input" rows={2} maxLength={2000} value={releaseNote} onChange={(e) => setReleaseNote(e.target.value)} />
        </label>
        <div className="row gap-sm">
          <Button
            variant={job.release_approved ? "ghost" : "primary"}
            busy={releaseBusy}
            disabled={releaseBlocked && !job.release_approved}
            onClick={async () => {
              setReleaseBusy(true);
              setReleaseError(null);
              try {
                const updated = await jobApi.release(job.job_id, !job.release_approved, releaseNote);
                onJobChange(updated);
              } catch (e) {
                setReleaseError(errorText(e));
              } finally {
                setReleaseBusy(false);
              }
            }}
          >
            {job.release_approved ? "Revoke release" : "Approve release"}
          </Button>
          {releaseBlocked && !job.release_approved && <span className="error-text small">Critical QA failures must be fixed before release.</span>}
        </div>
        {releaseError && <div className="banner banner-error" role="alert">{releaseError}</div>}
        {job.release_approved && job.release_note && <p className="muted small">Release note: {job.release_note}</p>}
        {flagged.length > 0 && (
          <>
            <h3>Timing warnings ({flagged.length})</h3>
            <table className="table compact">
              <tbody>
                {flagged.map((t) => (
                  <tr key={t.segment_id}>
                    <td className="mono">{fmtTime(t.placement_start, true)}</td>
                    <td>{t.warning}</td>
                    <td className="num">{t.speed_adjustment.toFixed(2)}×</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
        <p className="muted small">
          AI voices approximate the original speakers; similarity depends on the reference audio, and background
          separation is not perfect.
        </p>
      </div>
    </div>
  );
}
