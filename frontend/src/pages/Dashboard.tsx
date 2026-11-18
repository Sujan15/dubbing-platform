import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { projectApi } from "../api";
import { errorText } from "../api/client";
import type { ProjectSummary } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { Button, EmptyState, ErrorBanner, Field, Modal, Spinner, fmtDate } from "../components/ui";

export default function Dashboard() {
  const [query, setQuery] = useState("");
  const [creating, setCreating] = useState(false);
  const [renaming, setRenaming] = useState<ProjectSummary | null>(null);
  const projects = useAsync((s) => projectApi.list(undefined, s), [], {
    pollMs: 5000,
    shouldPoll: (d) => !!d?.some((p) => p.jobs.queued + p.jobs.running + p.jobs.generating > 0),
  });
  const navigate = useNavigate();
  const q = query.trim().toLowerCase();
  const items = (projects.data ?? []).filter(
    (p) => !q || p.name.toLowerCase().includes(q) || p.description.toLowerCase().includes(q),
  );

  return (
    <section className="page">
      <div className="page-head">
        <div>
          <h1>Projects</h1>
          <p className="muted">A project holds the episodes of one movie or series and its character voice library.</p>
        </div>
        <div className="row gap">
          <input
            className="input search"
            type="search"
            placeholder="Search projects"
            aria-label="Search projects"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <Button variant="primary" onClick={() => setCreating(true)}>
            New project
          </Button>
        </div>
      </div>

      <ErrorBanner error={projects.error} onRetry={projects.reload} />
      {projects.loading && !projects.data ? (
        <Spinner label="Loading projects" />
      ) : items.length === 0 ? (
        projects.data?.length ? (
          <EmptyState title="No matching projects">Try a different search.</EmptyState>
        ) : (
          <EmptyState
            title="No projects yet"
            action={
              <Button variant="primary" onClick={() => setCreating(true)}>
                Create your first project
              </Button>
            }
          >
            Create a project, upload an episode, review the Hindi draft, then generate the dub.
          </EmptyState>
        )
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>Project</th>
              <th className="num">Videos</th>
              <th className="num">Speakers</th>
              <th className="num">Completed</th>
              <th className="num">Pending review</th>
              <th className="num">In progress</th>
              <th className="num">Failed</th>
              <th>Last activity</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {items.map((p) => (
              <tr key={p.project_id} className="clickable" onClick={() => navigate(`/projects/${p.project_id}`)}>
                <td>
                  <Link to={`/projects/${p.project_id}`} onClick={(e) => e.stopPropagation()} className="strong">
                    {p.name}
                  </Link>
                  {p.description && <div className="muted small clamp">{p.description}</div>}
                </td>
                <td className="num">{p.video_count}</td>
                <td className="num">{p.speaker_count}</td>
                <td className="num">{p.jobs.succeeded}</td>
                <td className="num">{p.jobs.awaiting_review ? <span className="pill pill-review">{p.jobs.awaiting_review}</span> : 0}</td>
                <td className="num">{p.jobs.queued + p.jobs.running + p.jobs.generating}</td>
                <td className="num">{p.jobs.failed ? <span className="pill pill-failed">{p.jobs.failed}</span> : 0}</td>
                <td className="muted">{fmtDate(p.last_activity_at)}</td>
                <td>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={(e) => {
                      e.stopPropagation();
                      setRenaming(p);
                    }}
                  >
                    Rename
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {creating && (
        <ProjectForm
          title="New project"
          submitLabel="Create project"
          onClose={() => setCreating(false)}
          onSubmit={async (name, description) => {
            const p = await projectApi.create({ name, description });
            navigate(`/projects/${p.project_id}`);
          }}
        />
      )}
      {renaming && (
        <ProjectForm
          title="Rename project"
          submitLabel="Save"
          initialName={renaming.name}
          initialDescription={renaming.description}
          onClose={() => setRenaming(null)}
          onSubmit={async (name, description) => {
            await projectApi.update(renaming.project_id, { name, description });
            setRenaming(null);
            projects.reload();
          }}
        />
      )}
    </section>
  );
}

export function ProjectForm(props: {
  title: string;
  submitLabel: string;
  initialName?: string;
  initialDescription?: string;
  onClose: () => void;
  onSubmit: (name: string, description: string) => Promise<void>;
}) {
  const [name, setName] = useState(props.initialName ?? "");
  const [description, setDescription] = useState(props.initialDescription ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await props.onSubmit(name.trim(), description.trim());
    } catch (err) {
      setError(errorText(err));
      setBusy(false);
    }
  };
  return (
    <Modal title={props.title} onClose={props.onClose}>
      <form onSubmit={submit} className="stack">
        <Field label="Name">
          <input className="input" value={name} maxLength={120} onChange={(e) => setName(e.target.value)} required />
        </Field>
        <Field label="Description" hint="Optional — e.g. series, season, notes for the team.">
          <textarea className="input" rows={3} value={description} maxLength={2000} onChange={(e) => setDescription(e.target.value)} />
        </Field>
        <Field label="Target language">
          <input className="input" value="Hindi (hi)" disabled />
        </Field>
        {error && <div className="banner banner-error">{error}</div>}
        <div className="row end gap">
          <Button type="button" variant="ghost" onClick={props.onClose}>
            Cancel
          </Button>
          <Button type="submit" variant="primary" busy={busy} disabled={!name.trim()}>
            {props.submitLabel}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
