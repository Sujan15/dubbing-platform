// Test-only fixtures and a fetch router. Never imported by production code.
import { vi } from "vitest";
import type { DraftManifest, JobRecord } from "../api/types";

export type Handler = (req: { method: string; url: string; body: unknown }) => { status?: number; body?: unknown };

export function mockFetch(routes: Record<string, Handler>) {
  const calls: { method: string; url: string; body: unknown }[] = [];
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    const method = (init?.method ?? "GET").toUpperCase();
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ method, url, body });
    const key = Object.keys(routes).find((k) => {
      const [m, p] = k.split(" ");
      return m === method && new RegExp(`^${p}$`).test(url.split("?")[0]);
    });
    const res = key ? routes[key]({ method, url, body }) : { status: 404, body: { detail: `no route ${method} ${url}` } };
    const status = res.status ?? 200;
    return new Response(status === 204 ? null : JSON.stringify(res.body ?? {}), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fn);
  return calls;
}

export const job = (over: Partial<JobRecord> = {}): JobRecord => ({
  job_id: "j".repeat(32),
  state: "awaiting_review",
  stage: "awaiting_review",
  progress: 1,
  message: "Draft ready for review",
  input_filename: "episode01.mp4",
  options: { source_language: "en", mix_mode: "preserve_background", translation_engine: "openrouter", min_speakers: null, max_speakers: null },
  created_at: "2026-10-30T10:00:00Z",
  started_at: "2026-10-30T10:00:01Z",
  finished_at: null,
  source_language: "en",
  speakers: 2,
  segments: 2,
  warnings: [],
  error: null,
  project_id: "p".repeat(32),
  video_id: "v".repeat(32),
  review_required: true,
  phase: "analysis",
  approved_version: null,
  stage_history: [
    { stage: "transcribe", phase: "analysis", status: "completed", started_at: "2026-10-30T10:00:02Z", finished_at: "2026-10-30T10:01:02Z", message: "" },
  ],
  analysis_completed_at: null,
  generation_started_at: null,
  qa_status: null,
  qa_report_path: null,
  release_approved: false,
  release_note: "",
  engines: [],
  ...over,
});

export const draft = (over: Partial<DraftManifest> = {}): DraftManifest => ({
  manifest_version: 1,
  version: 3,
  job_id: "j".repeat(32),
  project_id: "p".repeat(32),
  video_id: "v".repeat(32),
  source_language: "en",
  target_language: "hi",
  pipeline_version: "2.0",
  media_duration: 10,
  mix_mode: "preserve_background",
  created_at: "2026-10-30T10:00:00Z",
  updated_at: "2026-10-30T10:00:00Z",
  review_status: "draft",
  approved_version: null,
  warnings: [],
  segments: [
    { segment_id: "seg_00001", start: 0.5, end: 2.8, original_text: "Hello there", translated_text: "नमस्ते", machine_translation: "नमस्ते", speaker_id: "SPEAKER_00", confidence: 0.9, review_status: "pending", warnings: [], speaker_display_name: "Rahul", voice_source: "cast", default_voice_id: null, library_speaker_id: null, reference_duration_seconds: 8 },
    { segment_id: "seg_00002", start: 3, end: 4, original_text: "I am fine", translated_text: "मैं ठीक हूँ", machine_translation: "मैं ठीक हूँ", speaker_id: "SPEAKER_01", confidence: 0.5, review_status: "pending", warnings: ["Low ASR confidence (0.50); check the original text."], speaker_display_name: "Speaker 2", voice_source: "clone", default_voice_id: null, library_speaker_id: null, reference_duration_seconds: 1 },
  ],
  speakers: [
    { speaker_id: "SPEAKER_00", display_name: "Rahul", color: 0, origin_labels: ["SPEAKER_00"], segment_count: 1, total_speech_duration: 2.3, usable_reference_duration: 8, reference_available: true, voice_source: "cast", library_speaker_id: null, default_voice_id: null, available_voice_options: ["cast", "character", "clone", "library"], warnings: [], engine: "parler", voice_id: null, character_id: null },
    { speaker_id: "SPEAKER_01", display_name: "Speaker 2", color: 1, origin_labels: ["SPEAKER_01"], segment_count: 1, total_speech_duration: 1, usable_reference_duration: 1, reference_available: true, voice_source: "cast", library_speaker_id: null, default_voice_id: null, available_voice_options: ["cast", "character", "clone", "library"], warnings: ["Only 1.0s of clean speech (minimum 6s): the clone will be less faithful; consider a library or default voice."], engine: "parler", voice_id: null, character_id: null },
  ],
  ...over,
});
