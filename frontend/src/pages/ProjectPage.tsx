import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { projectApi } from "../api";
import { errorText } from "../api/client";
import type { JobRecord, Meta } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import SpeakerLibrary from "../components/SpeakerLibrary";
import CharacterLibrary from "../components/CharacterLibrary";
import UploadPanel from "../components/UploadPanel";
import { Button, EmptyState, ErrorBanner, Modal, Spinner, StatusBadge, STAGE_LABEL, fmtBytes, fmtDate, fmtTime } from "../components/ui";
import { ProjectForm } from "./Dashboard";

type Tab = "videos" | "speakers" | "characters";

export function jobAction(job: JobRecord | null): { to: string; label: string } | null {
  if (!job) return null;
  switch (job.state) {
    case "awaiting_review":
      return { to: `/jobs/${job.job_id}/review`, label: "Review" };
    case "succeeded":
      return { to: `/jobs/${job.job_id}`, label: "Open result" };
    default:
      return { to: `/jobs/${job.job_id}`, label: job.state === "failed" || job.state === "cancelled" ? "Details" : "Progress" };
  }
}

export default function ProjectPage({ meta }: { meta?: Meta }) {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const [tab, setTab] = useState<Tab>("videos");
  const [showUpload, setShowUpload] = useState(false);
  const [editing, setEditing] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const project = useAsync((s) => projectApi.get(projectId, s), [projectId], {
    pollMs: 4000,
    shouldPoll: (d) => !!d?.videos.some((v) => v.job && ["queued", "running", "generating"].includes(v.job.state)),
  });

  if (project.loading && !project.data) return <Spinner label="Loading project" />;
  if (!project.data) return <ErrorBanner error={project.error ?? "Project not found"} onRetry={project.reload} />;
  const p = project.data;
  const pending = p.videos.filter((v) => v.job?.state === "awaiting_review");

  return (
    <section className="page">
      <div className="crumbs">
        <Link to="/">Projects</Link> / <span>{p.name}</span>
      </div>
      <div className="page-head">
        <div>
          <h1>{p.name}</h1>
          {p.description && <p className="muted">{p.description}</p>}
          <div className="stats">
            <span><b>{p.video_count}</b> videos</span>
            <span><b>{p.speaker_count}</b> library speakers</span>
            <span><b>{p.jobs.succeeded}</b> completed</span>
            <span><b>{p.jobs.awaiting_review}</b> pending review</span>
            <span><b>{p.jobs.failed}</b> failed</span>
            <span className="muted">Target: Hindi · updated {fmtDate(p.updated_at)}</span>
          </div>
        </div>
        <div className="row gap">
          <Button variant="ghost" onClick={() => setEditing(true)}>Edit</Button>
          <Button variant="ghost" onClick={() => setDeleting(true)}>Delete</Button>
          <Button variant="primary" onClick={() => { setTab("videos"); setShowUpload(true); }}>
            Upload video
          </Button>
        </div>
      </div>
      <ErrorBanner error={project.error} onRetry={project.reload} />

      {pending.length > 0 && (
        <div className="banner banner-info">
          <span>
            {pending.length} episode{pending.length > 1 ? "s" : ""} waiting for review:{" "}
            {pending.map((v, i) => (
              <span key={v.video_id}>
                {i > 0 && ", "}
                <Link to={`/jobs/${v.job!.job_id}/review`}>{v.title}</Link>
              </span>
            ))}
          </span>
        </div>
      )}

      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === "videos"} className={tab === "videos" ? "active" : ""} onClick={() => setTab("videos")}>
          Videos ({p.videos.length})
        </button>
        <button role="tab" aria-selected={tab === "speakers"} className={tab === "speakers" ? "active" : ""} onClick={() => setTab("speakers")}>
          Speaker Library ({p.speakers.length})
        </button>
        <button role="tab" aria-selected={tab === "characters"} className={tab === "characters" ? "active" : ""} onClick={() => setTab("characters")}>
          Characters
        </button>
      </div>

      {tab === "videos" && (
        <div className="stack-lg">
          {showUpload && (
            <UploadPanel
              project={p}
              meta={meta}
              onUploaded={(r) => {
                setShowUpload(false);
                project.reload();
                navigate(`/jobs/${r.job_id}`);
              }}
            />
          )}
          {p.videos.length === 0 ? (
            !showUpload && (
              <EmptyState title="No videos in this project" action={<Button variant="primary" onClick={() => setShowUpload(true)}>Upload the first episode</Button>}>
                Each upload is analysed (transcription, speakers, Hindi draft) and then waits for your review before any voice is generated.
              </EmptyState>
            )
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Video</th>
                  <th className="num">Duration</th>
                  <th className="num">Size</th>
                  <th>Status</th>
                  <th>Stage</th>
                  <th className="num">Speakers</th>
                  <th className="num">Lines</th>
                  <th>Uploaded</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {p.videos.map((v) => {
                  const action = jobAction(v.job);
                  return (
                    <tr key={v.video_id}>
                      <td>
                        <div className="strong">{v.title}</div>
                        <div className="muted small">{v.original_filename}</div>
                      </td>
                      <td className="num">{fmtTime(v.duration)}</td>
                      <td className="num">{fmtBytes(v.size_bytes)}</td>
                      <td>{v.job ? <StatusBadge state={v.job.state} /> : "—"}</td>
                      <td className="muted small">
                        {v.job && ["queued", "running", "generating"].includes(v.job.state) ? STAGE_LABEL[v.job.stage] ?? v.job.stage : v.job?.error ? <span className="error-text clamp" title={v.job.error}>{v.job.error}</span> : "—"}
                      </td>
                      <td className="num">{v.job?.speakers ?? "—"}</td>
                      <td className="num">{v.job?.segments ?? "—"}</td>
                      <td className="muted small">{fmtDate(v.created_at)}</td>
                      <td>{action && <Link className="btn btn-sm btn-secondary" to={action.to}>{action.label}</Link>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}

      {tab === "speakers" && <SpeakerLibrary projectId={p.project_id} speakers={p.speakers} meta={meta} onChanged={project.reload} />}
      {tab === "characters" && <CharacterLibrary projectId={p.project_id} />}

      {editing && (
        <ProjectForm
          title="Edit project"
          submitLabel="Save"
          initialName={p.name}
          initialDescription={p.description}
          onClose={() => setEditing(false)}
          onSubmit={async (name, description) => {
            await projectApi.update(p.project_id, { name, description });
            setEditing(false);
            project.reload();
          }}
        />
      )}
      {deleting && (
        <Modal
          title={`Delete “${p.name}”?`}
          onClose={() => setDeleting(false)}
          footer={
            <>
              <Button variant="ghost" onClick={() => setDeleting(false)}>Cancel</Button>
              <Button
                variant="danger"
                onClick={async () => {
                  try {
                    await projectApi.remove(p.project_id, true);
                    navigate("/");
                  } catch (e) {
                    setDeleteError(errorText(e));
                  }
                }}
              >
                Delete project and all files
              </Button>
            </>
          }
        >
          <p>
            This permanently deletes {p.video_count} uploaded video(s), their drafts and dubbed outputs, and {p.speaker_count} library speaker(s).
            Projects with jobs in progress cannot be deleted.
          </p>
          {deleteError && <div className="banner banner-error">{deleteError}</div>}
        </Modal>
      )}
    </section>
  );
}
