import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import Dashboard from "../pages/Dashboard";
import JobPage from "../pages/JobPage";
import ProjectPage from "../pages/ProjectPage";
import ReviewPage, { applyPending } from "../pages/ReviewPage";
import VoiceLibraryPage from "../pages/VoiceLibraryPage";
import CharacterLibrary from "../components/CharacterLibrary";
import StageList from "../components/StageList";
import { draft, job, mockFetch } from "./fixtures";

afterEach(() => vi.unstubAllGlobals());

const P = "p".repeat(32);
const J = "j".repeat(32);

function at(path: string, element: React.ReactNode, route: string) {
  return render(
    <MemoryRouter initialEntries={[path]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes>
        <Route path={route} element={element} />
        <Route path="*" element={<div>navigated</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

const summary = (over = {}) => ({
  project_id: P, name: "My Hindi Series", description: "Season 1", target_language: "hi",
  settings: { source_language: null, mix_mode: "preserve_background", translation_engine: null, review_enabled: null },
  status: "active", created_at: "2026-10-30T10:00:00Z", updated_at: "2026-10-30T10:00:00Z",
  video_count: 2, speaker_count: 3,
  jobs: { queued: 0, running: 0, awaiting_review: 1, generating: 0, succeeded: 1, failed: 0, cancelled: 0 },
  last_activity_at: "2026-10-30T10:00:00Z", ...over,
});

describe("Project dashboard", () => {
  it("shows the empty state and creates a project", async () => {
    const calls = mockFetch({
      "GET /api/v1/projects": () => ({ body: [] }),
      "POST /api/v1/projects": ({ body }) => ({ status: 201, body: summary({ name: (body as { name: string }).name }) }),
    });
    at("/", <Dashboard />, "/");
    expect(await screen.findByText("No projects yet")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Create your first project" }));
    await userEvent.type(screen.getByLabelText("Name"), "Episode Pack");
    await userEvent.click(screen.getByRole("button", { name: "Create project" }));
    await screen.findByText("navigated");
    expect(calls.find((c) => c.method === "POST")?.body).toEqual({ name: "Episode Pack", description: "" });
  });

  it("lists projects with review counts and filters by search", async () => {
    mockFetch({ "GET /api/v1/projects": () => ({ body: [summary(), summary({ project_id: "q".repeat(32), name: "Other" })] }) });
    at("/", <Dashboard />, "/");
    expect(await screen.findByText("My Hindi Series")).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText("Search projects"), "other");
    expect(screen.queryByText("My Hindi Series")).not.toBeInTheDocument();
    expect(screen.getByText("Other")).toBeInTheDocument();
  });

  it("shows an error state when the API fails", async () => {
    mockFetch({ "GET /api/v1/projects": () => ({ status: 500, body: { detail: "database unavailable" } }) });
    at("/", <Dashboard />, "/");
    expect(await screen.findByRole("alert")).toHaveTextContent("database unavailable");
  });
});

describe("Project detail + upload", () => {
  it("rejects unsupported files before uploading", async () => {
    mockFetch({
      "GET /api/v1/projects/p+": () => ({ body: { ...summary(), videos: [], speakers: [] } }),
      "GET /api/v1/default-voices": () => ({ body: { voices: [], invalid: [] } }),
    });
    at(`/projects/${P}`, <ProjectPage />, "/projects/:projectId");
    await userEvent.click(await screen.findByRole("button", { name: "Upload the first episode" }));
    const input = screen.getByTestId("video-input") as HTMLInputElement;
    await userEvent.upload(input, new File(["x"], "notes.txt", { type: "text/plain" }), { applyAccept: false });
    expect(await screen.findByRole("alert")).toHaveTextContent('Unsupported file type ".txt"');
    expect(screen.getByRole("button", { name: "Upload & analyze" })).toBeDisabled();
  });

  it("shows the speaker library empty state and default voice bank", async () => {
    mockFetch({
      "GET /api/v1/projects/p+": () => ({ body: { ...summary(), videos: [], speakers: [] } }),
      "GET /api/v1/default-voices": () => ({ body: { voices: [], invalid: [] } }),
    });
    at(`/projects/${P}`, <ProjectPage />, "/projects/:projectId");
    await userEvent.click(await screen.findByRole("tab", { name: /Speaker Library/ }));
    expect(screen.getByText("No library speakers yet")).toBeInTheDocument();
    expect(await screen.findByText("No default voices installed")).toBeInTheDocument();
  });
});

describe("Review workstation", () => {
  function routes(overrides: Record<string, Parameters<typeof mockFetch>[0][string]> = {}) {
    return mockFetch({
      "GET /api/v1/jobs/j+": () => ({ body: job() }),
      "GET /api/v1/jobs/j+/draft": () => ({ body: draft() }),
      "GET /api/v1/projects/p+/speakers": () => ({ body: [] }),
      "GET /api/v1/default-voices": () => ({
        body: { voices: [{ voice_id: "narrator", label: "Narrator", description: "", gender: null, duration_seconds: 18, speech_seconds: 17, sample_rate: 22050, valid: true, problem: null }], invalid: [] },
      }),
      "GET /api/v1/engines": () => ({ body: { engines: [], enabled: ["parler", "indicf5"], default_engine: "parler", default_clone_engine: "indicf5" } }),
      "GET /api/v1/voice-library": () => ({ body: { voices: [], engines: [] } }),
      "GET /api/v1/projects/p+/characters": () => ({ body: [] }),
      "PATCH /api/v1/jobs/j+/draft": ({ body }) => ({ body: draft({ version: (body as { base_version: number }).base_version + 1 }) }),
      ...overrides,
    });
  }

  it("applies unsaved edits locally", () => {
    const d = applyPending(draft(), { segments: { seg_00002: { segment_id: "seg_00002", speaker_id: "SPEAKER_00" } }, speakers: { SPEAKER_00: { speaker_id: "SPEAKER_00", display_name: "Rahul K" } } });
    expect(d.segments[1].speaker_display_name).toBe("Rahul K");
    expect(d.speakers[0].display_name).toBe("Rahul K");
  });

  it("edits Hindi text and selects an explicit IndicF5 source, then saves one versioned PATCH", async () => {
    const calls = routes();
    at(`/jobs/${J}/review`, <ReviewPage />, "/jobs/:jobId/review");
    const row = await screen.findByTestId("seg-seg_00001");
    const hindi = within(row).getByLabelText("Hindi translation");
    await userEvent.clear(hindi);
    await userEvent.type(hindi, "नमस्ते दोस्त");
    expect(within(row).getByText(/Machine:/)).toBeInTheDocument();

    const group = screen.getByRole("radiogroup", { name: "Voice source for Speaker 2" });
    await userEvent.click(within(group).getByLabelText("IndicF5 from this video"));
    expect(screen.getByText("2 unsaved changes")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() => expect(calls.some((c) => c.method === "PATCH")).toBe(true));
    const patch = calls.find((c) => c.method === "PATCH")!.body as Record<string, unknown>;
    expect(patch.base_version).toBe(3);
    expect(patch.segments).toEqual([{ segment_id: "seg_00001", translated_text: "नमस्ते दोस्त" }]);
    expect(patch.speakers).toEqual([{ speaker_id: "SPEAKER_01", voice_source: "clone", engine: "indicf5" }]);
  });

  it("reassigns a line to another speaker", async () => {
    const calls = routes();
    at(`/jobs/${J}/review`, <ReviewPage />, "/jobs/:jobId/review");
    const row = await screen.findByTestId("seg-seg_00002");
    await userEvent.selectOptions(within(row).getByLabelText("Speaker"), "SPEAKER_00");
    await userEvent.click(screen.getByRole("button", { name: "Save draft" }));
    await waitFor(() => expect(calls.some((c) => c.method === "PATCH")).toBe(true));
    expect((calls.find((c) => c.method === "PATCH")!.body as { segments: unknown[] }).segments).toEqual([
      { segment_id: "seg_00002", speaker_id: "SPEAKER_00" },
    ]);
  });

  it("shows a reload prompt on a version conflict", async () => {
    routes({ "PATCH /api/v1/jobs/j+/draft": () => ({ status: 409, body: { detail: "Draft was modified" } }) });
    at(`/jobs/${J}/review`, <ReviewPage />, "/jobs/:jobId/review");
    const row = await screen.findByTestId("seg-seg_00001");
    await userEvent.type(within(row).getByLabelText("Hindi translation"), "!");
    await userEvent.click(screen.getByRole("button", { name: "Save draft" }));
    expect(await screen.findByRole("button", { name: "Reload draft" })).toBeInTheDocument();
  });

  it("blocks confirmation while validation errors exist", async () => {
    routes({
      "GET /api/v1/jobs/j+/confirm-preview": () => ({
        body: { version: 3, errors: ["seg_00002: Hindi translation is empty."], warnings: [], speakers: [] },
      }),
    });
    at(`/jobs/${J}/review`, <ReviewPage />, "/jobs/:jobId/review");
    await userEvent.click(await screen.findByRole("button", { name: "Confirm & Generate" }));
    expect(await screen.findByText("seg_00002: Hindi translation is empty.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Lock draft v3/ })).toBeDisabled();
  });

  it("confirms with the current version and opens the job", async () => {
    const calls = routes({
      "GET /api/v1/jobs/j+/confirm-preview": () => ({
        body: { version: 3, errors: [], warnings: ["Speaker 2: short reference"], speakers: [{ speaker_id: "SPEAKER_00", display_name: "Rahul", voice_source: "clone", library_speaker_id: null, default_voice_id: null, segment_count: 1 }] },
      }),
      "POST /api/v1/jobs/j+/confirm": () => ({ status: 202, body: { job: job({ state: "generating" }), approved_version: 3, warnings: [] } }),
    });

    at(`/jobs/${J}/review`, <ReviewPage />, "/jobs/:jobId/review");
    await userEvent.click(await screen.findByRole("button", { name: "Confirm & Generate" }));
    await userEvent.click(await screen.findByRole("button", { name: /Lock draft v3/ }));
    await screen.findByText("navigated");
    expect(calls.find((c) => c.method === "POST")!.body).toEqual({ base_version: 3 });
  });

  it("confirms an identity suggestion without changing the selected voice source", async () => {
    const suggested = draft({
      speakers: draft().speakers.map((speaker, index) => index === 0 ? {
        ...speaker,
        mapping_status: "suggested",
        suggested_character_id: "c".repeat(32),
        suggested_character_name: "Asha",
        identity_confidence: 0.91,
        identity_reason: "voice similarity",
      } : speaker),
    });
    const calls = routes({
      "GET /api/v1/jobs/j+/draft": () => ({ body: suggested }),
      "GET /api/v1/projects/p+/characters": () => ({
        body: [{
          character_id: "c".repeat(32), series_id: P, canonical_name: "Asha", aliases: [], description: "",
          role: "heroine", gender: "female", status: "active", voice_profiles: [],
          active_voice_profile_id: null, has_approved_voice: false, episodes: [], embedding_count: 1,
        }],
      }),
    });
    at(`/jobs/${J}/review`, <ReviewPage />, "/jobs/:jobId/review");
    await userEvent.click(await screen.findByRole("button", { name: "Confirm identity" }));
    await userEvent.click(screen.getByRole("button", { name: "Save draft" }));
    await waitFor(() => expect(calls.some((call) => call.method === "PATCH")).toBe(true));
    const patch = calls.find((call) => call.method === "PATCH")!.body as { speakers: Record<string, unknown>[] };
    expect(patch.speakers).toEqual([{
      speaker_id: "SPEAKER_00",
      character_id: "c".repeat(32),
      reject_suggestion: false,
    }]);
  });

  it("is read-only once approved", async () => {
    routes({
      "GET /api/v1/jobs/j+": () => ({ body: job({ state: "generating", phase: "generation", approved_version: 3 }) }),
      "GET /api/v1/jobs/j+/draft": () => ({ body: draft({ review_status: "approved", approved_version: 3 }) }),
    });
    at(`/jobs/${J}/review`, <ReviewPage />, "/jobs/:jobId/review");
    expect(await screen.findByText(/Approved draft v3 is locked/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Confirm & Generate" })).not.toBeInTheDocument();
    expect(within(screen.getByTestId("seg-seg_00001")).getByLabelText("Hindi translation")).toHaveAttribute("readonly");
  });
});

describe("Generation progress and result", () => {
  it("renders real stage states without percentages", () => {
    render(
      <StageList
        job={job({
          state: "generating", phase: "generation", stage: "synthesize", message: "Hindi line 3/19",
          stage_history: [
            { stage: "resolve_voices", phase: "generation", status: "completed", started_at: "2026-10-30T10:00:00Z", finished_at: "2026-10-30T10:00:02Z", message: "" },
            { stage: "synthesize", phase: "generation", status: "running", started_at: "2026-10-30T10:00:02Z", finished_at: null, message: "" },
          ],
        })}
        phase="generation"
      />,
    );
    expect(screen.getByText("Hindi line 3/19")).toBeInTheDocument();
    expect(screen.getByText("Done · 2s")).toBeInTheDocument();
    expect(screen.getAllByText("Pending").length).toBeGreaterThan(0);
    expect(document.body.textContent).not.toMatch(/\d+%/);
  });

  it("shows the final player, download and voice mapping", async () => {
    mockFetch({
      "GET /api/v1/jobs/j+": () => ({ body: job({ state: "succeeded", phase: "generation", finished_at: "2026-10-30T10:30:00Z", warnings: ["preserve_background needs separated stems"] }) }),
      "GET /api/v1/jobs/j+/report": () => ({
        body: { source_language: "en", target_language: "hi", mix_mode_requested: "preserve_background", mix_mode_used: "original_plus_dub", separation: false, speakers: 1, segments: 2, warnings: [], runtime: {},
          voices: { SPEAKER_00: { speaker_id: "SPEAKER_00", display_name: "Rahul", requested_source: "clone", resolved_source: "clone", reference_seconds: 9.5, library_speaker_id: null, default_voice_id: null, fallback_applied: false, warnings: [] } } },
      }),
      "GET /api/v1/jobs/j+/timing-report": () => ({ body: { summary: { fits: 2 }, segments: [] } }),
      "GET /api/v1/jobs/j+/qa-report": () => ({ body: { status: "PASS", checks: {}, metrics: {}, clipping: { clipped_samples: 0, peak: 0.9 }, release_blocked: false, note: "Listener review required." } }),
    });
    const { container } = at(`/jobs/${J}`, <JobPage />, "/jobs/:jobId");
    expect(await screen.findByRole("link", { name: "Download dubbed video" })).toHaveAttribute("href", `/api/v1/jobs/${J}/result`);
    expect(container.querySelector("video")).toHaveAttribute("src", `/api/v1/jobs/${J}/output-video`);
    expect(await screen.findByText("Rahul")).toBeInTheDocument();
    expect(screen.getByText(/original_plus_dub \(requested preserve_background\)/)).toBeInTheDocument();
    expect(screen.getByText("preserve_background needs separated stems")).toBeInTheDocument();
  });

  it("blocks release when critical QA fails", async () => {
    mockFetch({
      "GET /api/v1/jobs/j+": () => ({ body: job({ state: "succeeded", phase: "generation", qa_status: "FAILED" }) }),
      "GET /api/v1/jobs/j+/report": () => ({ body: { voices: {}, warnings: [], mix_mode_requested: "preserve_background", mix_mode_used: "preserve_background" } }),
      "GET /api/v1/jobs/j+/timing-report": () => ({ body: { summary: {}, segments: [] } }),
      "GET /api/v1/jobs/j+/qa-report": () => ({
        body: {
          status: "FAILED", checks: { clipping: { status: "FAILED", value: 12, detail: "clipped samples" } },
          metrics: {}, clipping: { clipped_samples: 12, peak: 1.5 }, release_blocked: true, note: "Fix critical checks.",
        },
      }),
    });
    at(`/jobs/${J}`, <JobPage />, "/jobs/:jobId");
    expect(await screen.findByRole("button", { name: "Approve release" })).toBeDisabled();
    expect(screen.getByText("Critical QA failures must be fixed before release.")).toBeInTheDocument();
  });

  it("records a release approval after QA passes", async () => {
    const calls = mockFetch({
      "GET /api/v1/jobs/j+": () => ({ body: job({ state: "succeeded", phase: "generation", qa_status: "PASS" }) }),
      "GET /api/v1/jobs/j+/report": () => ({ body: { voices: {}, warnings: [], mix_mode_requested: "preserve_background", mix_mode_used: "preserve_background" } }),
      "GET /api/v1/jobs/j+/timing-report": () => ({ body: { summary: {}, segments: [] } }),
      "GET /api/v1/jobs/j+/qa-report": () => ({
        body: { status: "PASS", checks: {}, metrics: {}, clipping: { clipped_samples: 0, peak: 0.8 }, release_blocked: false, note: "" },
      }),
      "POST /api/v1/jobs/j+/release": ({ body }) => ({
        body: job({ state: "succeeded", phase: "generation", qa_status: "PASS", release_approved: (body as { approved: boolean }).approved, release_note: (body as { note: string }).note }),
      }),
    });
    at(`/jobs/${J}`, <JobPage />, "/jobs/:jobId");
    await userEvent.click(await screen.findByRole("button", { name: "Approve release" }));
    expect(await screen.findByText("Released")).toBeInTheDocument();
    expect(calls.find((call) => call.method === "POST")?.body).toEqual({ approved: true, note: "" });
  });
});

describe("Voice and character libraries", () => {
  const profile = (overrides: Record<string, unknown> = {}) => ({
    voice_id: "v".repeat(32), voice_name: "Rohit", kind: "cast", engine: "parler", language: "hi",
    gender: "male", description: "Hindi dialogue", catalog_slot: null, reference_available: false,
    reference_transcript: null, reference_language: "hi", reference_duration_seconds: 0,
    reference_speech_seconds: 0, reference_sample_rate: null, reference_quality: [],
    voice_description_prompt: "A clear, natural Hindi male voice.", prompt_version: 1,
    prompt_versions: [{ version: 1, prompt: "A clear, natural Hindi male voice.", created_at: "2026-10-30T10:00:00Z" }],
    sample_text: "नमस्ते।", generation_parameters: {}, seed: null, model_id: "ai4bharat/indic-parler-tts",
    model_revision: null, supported_language_status: "unvalidated", license_metadata: {}, quality_benchmark: {},
    approved: false, approved_at: null, approved_config_hash: null, approval_note: "", active: true,
    preview: null, preview_url: null, selectable: false, needs_reaudition: false, origin: {},
    created_at: "2026-10-30T10:00:00Z", updated_at: "2026-10-30T10:00:00Z", ...overrides,
  });

  it("auditions a backend profile and approves it only after the audition completes", async () => {
    const calls = mockFetch({
      "GET /api/v1/engines": () => ({
        body: { engines: [{ engine: "parler", enabled: true, available: true, reason: "", model_id: "ai4bharat/indic-parler-tts", model_revision: null, capabilities: { reference_conditioned: false, needs_reference_transcript: false, prompt_conditioned: true, supports_seed: true, supports_speed: false, languages: ["hi"] }, license: "Apache-2.0", commercial_use: null, legacy: false }], enabled: ["parler"], default_engine: "parler", default_clone_engine: "indicf5" },
      }),
      "GET /api/v1/voice-library": () => ({ body: { voices: [profile()], engines: [] } }),
      "POST /api/v1/voice-library/v+/audition": () => ({ status: 202, body: { audition_id: "r".repeat(32), voice_id: "v".repeat(32), text: "नमस्ते।", status: "queued", error: null, created_at: "", finished_at: null } }),
      "GET /api/v1/voice-library/auditions/r+": () => ({ body: { audition_id: "r".repeat(32), voice_id: "v".repeat(32), text: "नमस्ते।", status: "done", error: null, problems: [], created_at: "", finished_at: "" } }),
      "POST /api/v1/voice-library/v+/approval": ({ body }) => ({ body: profile({ approved: (body as { approved: boolean }).approved, selectable: true }) }),
    });
    at("/voices", <VoiceLibraryPage />, "/voices");
    expect(await screen.findByText("Rohit")).toBeInTheDocument();
    expect(screen.getAllByText("parler").length).toBeGreaterThan(0);
    const audition = screen.getByRole("button", { name: "Audition" });
    await userEvent.click(audition);
    await screen.findByLabelText("Rohit audition");
    await userEvent.click(screen.getByRole("button", { name: "Approve after audition" }));
    await waitFor(() => expect(calls.some((call) => call.method === "POST" && call.url.includes("/approval"))).toBe(true));
  });

  it("creates a character and assigns the selected approved profile", async () => {
    const calls = mockFetch({
      "GET /api/v1/projects/p+/characters": () => ({ body: [] }),
      "GET /api/v1/voice-library": () => ({ body: { voices: [profile({ approved: true, selectable: true })], engines: [] } }),
      "POST /api/v1/projects/p+/characters": () => ({
        status: 201,
        body: { character_id: "c".repeat(32), series_id: P, canonical_name: "Asha", aliases: [], description: "", role: "heroine", gender: "female", status: "active", voice_profiles: [], active_voice_profile_id: null, has_approved_voice: false, episodes: [], embedding_count: 0 },
      }),
      "POST /api/v1/projects/p+/characters/c+/voice-profiles": ({ body }) => ({
        status: 201,
        body: { ...JSON.parse(JSON.stringify(body)), character_id: "c".repeat(32), series_id: P, canonical_name: "Asha", aliases: [], description: "", role: "heroine", gender: "female", status: "active", voice_profiles: [], active_voice_profile_id: "vp", has_approved_voice: true, episodes: [], embedding_count: 0 },
      }),
    });
    at(`/projects/${P}`, <CharacterLibrary projectId={P} />, "/projects/:projectId");
    await userEvent.type(await screen.findByLabelText("Character name"), "Asha");
    await userEvent.selectOptions(screen.getByLabelText("Role"), "heroine");
    await userEvent.selectOptions(screen.getByLabelText("Gender"), "female");
    await userEvent.selectOptions(screen.getByLabelText("Approved voice profile (optional)"), "v".repeat(32));
    await userEvent.click(screen.getByRole("button", { name: "Create character" }));
    await waitFor(() => expect(calls.some((call) => call.method === "POST" && call.url.includes("/voice-profiles"))).toBe(true));
    expect(calls.find((call) => call.method === "POST" && call.url.endsWith("/characters"))?.body).toMatchObject({
      canonical_name: "Asha", role: "heroine", gender: "female",
    });
  });
});
