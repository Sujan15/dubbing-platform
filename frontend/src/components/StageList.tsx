import type { JobRecord } from "../api/types";
import { STAGE_LABEL, fmtDuration } from "./ui";

export const ANALYSIS_ORDER = ["validate", "extract", "separate", "transcribe", "diarize", "align", "translate", "voices"];
export const GENERATION_ORDER = ["prepare", "voices", "resolve_voices", "synthesize", "mix", "mux", "validate_output"];
const SINGLE_SHOT_ORDER = [...ANALYSIS_ORDER, "resolve_voices", "synthesize", "mix", "mux", "validate_output"];

/** Real stage states from the job's stage_history; no invented percentages. */
export default function StageList({ job, phase }: { job: JobRecord; phase: "analysis" | "generation" }) {
  const order = phase === "generation" ? GENERATION_ORDER : job.review_required ? ANALYSIS_ORDER : SINGLE_SHOT_ORDER;
  const events = job.stage_history.filter((e) => e.phase === phase);
  const byStage = new Map(events.map((e) => [e.stage, e]));
  const active = ["queued", "running", "generating"].includes(job.state);
  return (
    <ol className="stages">
      {order.map((stage) => {
        const ev = byStage.get(stage);
        const status = ev ? ev.status : "pending";
        const isCurrent = active && status === "running";
        return (
          <li key={stage} className={`stage stage-${status}`} aria-current={isCurrent ? "step" : undefined}>
            <span className="stage-dot" aria-hidden />
            <span className="stage-name">{STAGE_LABEL[stage] ?? stage}</span>
            <span className="stage-meta">
              {status === "pending" && "Pending"}
              {status === "running" && (job.stage === stage ? job.message || "Running" : "Running")}
              {status === "completed" && `Done · ${fmtDuration(ev!.started_at, ev!.finished_at)}`}
              {status === "failed" && "Failed"}
              {status === "cancelled" && "Cancelled"}
            </span>
            {status === "running" && <span className="stage-elapsed">{fmtDuration(ev!.started_at)}</span>}
          </li>
        );
      })}
    </ol>
  );
}
