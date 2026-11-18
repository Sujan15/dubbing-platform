import { API_BASE, request, upload } from "./client";
import type {
  ConfirmPreview,
  DefaultVoiceList,
  DraftManifest,
  DraftPatch,
  EngineCatalog,
  JobRecord,
  JobReport,
  LibrarySpeaker,
  Meta,
  MixMode,
  QaReport,
  ProjectDetail,
  ProjectSettings,
  ProjectSummary,
  ProjectVideo,
  ResolvedVoice,
  SeriesCharacter,
  TimingReport,
  VideoUploadResponse,
  VoiceAudition,
  VoiceProfile,
  VoiceProfileList,
} from "./types";

const enc = encodeURIComponent;

export const metaApi = {
  get: (signal?: AbortSignal) => request<Meta>("/meta", { signal }),
};

export const projectApi = {
  list: (q?: string, signal?: AbortSignal) =>
    request<ProjectSummary[]>(`/projects${q ? `?q=${enc(q)}` : ""}`, { signal }),
  get: (id: string, signal?: AbortSignal) => request<ProjectDetail>(`/projects/${enc(id)}`, { signal }),
  create: (body: { name: string; description?: string; settings?: Partial<ProjectSettings> }) =>
    request<ProjectSummary>("/projects", { method: "POST", json: body }),
  update: (id: string, body: { name?: string; description?: string; settings?: ProjectSettings }) =>
    request<ProjectSummary>(`/projects/${enc(id)}`, { method: "PATCH", json: body }),
  remove: (id: string, force = false) =>
    request<void>(`/projects/${enc(id)}?force=${force}`, { method: "DELETE" }),
};

export interface UploadOptions {
  title?: string;
  source_language?: string;
  mix_mode?: MixMode;
  min_speakers?: number;
  max_speakers?: number;
}

export const videoApi = {
  list: (projectId: string) => request<ProjectVideo[]>(`/projects/${enc(projectId)}/videos`),
  upload: (projectId: string, file: File, opts: UploadOptions, onProgress?: (f: number) => void) => {
    const form = new FormData();
    form.append("file", file);
    Object.entries(opts).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") form.append(k, String(v));
    });
    return upload<VideoUploadResponse>(`/projects/${enc(projectId)}/videos`, form, onProgress);
  },
};

export const jobApi = {
  get: (id: string, signal?: AbortSignal) => request<JobRecord>(`/jobs/${enc(id)}`, { signal }),
  list: (projectId?: string) => request<JobRecord[]>(`/jobs${projectId ? `?project_id=${enc(projectId)}` : ""}`),
  cancel: (id: string) => request<JobRecord>(`/jobs/${enc(id)}/cancel`, { method: "POST" }),
  retry: (id: string) => request<JobRecord>(`/jobs/${enc(id)}/retry`, { method: "POST" }),
  logs: (id: string, tail = 40, signal?: AbortSignal) =>
    request<{ lines: string[]; total_lines: number }>(`/jobs/${enc(id)}/logs?tail=${tail}`, { signal }),
  release: (id: string, approved: boolean, note = "") =>
    request<JobRecord>(`/jobs/${enc(id)}/release`, { method: "POST", json: { approved, note } }),
};

export const draftApi = {
  get: (jobId: string, signal?: AbortSignal) => request<DraftManifest>(`/jobs/${enc(jobId)}/draft`, { signal }),
  patch: (jobId: string, patch: DraftPatch) =>
    request<DraftManifest>(`/jobs/${enc(jobId)}/draft`, { method: "PATCH", json: patch }),
  preview: (jobId: string) => request<ConfirmPreview>(`/jobs/${enc(jobId)}/confirm-preview`),
  confirm: (jobId: string, baseVersion: number) =>
    request<{ job: JobRecord; approved_version: number; warnings: string[] }>(`/jobs/${enc(jobId)}/confirm`, {
      method: "POST",
      json: { base_version: baseVersion },
    }),
};

export const speakerApi = {
  list: (projectId: string) => request<LibrarySpeaker[]>(`/projects/${enc(projectId)}/speakers`),
  create: (projectId: string, body: { display_name: string; notes?: string }) =>
    request<LibrarySpeaker>(`/projects/${enc(projectId)}/speakers`, { method: "POST", json: body }),
  update: (projectId: string, id: string, body: { display_name?: string; notes?: string; reference_transcript?: string }) =>
    request<LibrarySpeaker>(`/projects/${enc(projectId)}/speakers/${enc(id)}`, { method: "PATCH", json: body }),
  remove: (projectId: string, id: string) =>
    request<void>(`/projects/${enc(projectId)}/speakers/${enc(id)}`, { method: "DELETE" }),
  uploadReference: (projectId: string, id: string, file: File, replace: boolean, transcript: string,
    onProgress?: (f: number) => void) => {
    const form = new FormData();
    form.append("file", file);
    form.append("replace", String(replace));
    form.append("transcript", transcript);
    return upload<LibrarySpeaker>(`/projects/${enc(projectId)}/speakers/${enc(id)}/reference`, form, onProgress);
  },
  promote: (projectId: string, body: { job_id: string; job_speaker_id: string; display_name?: string }) =>
    request<LibrarySpeaker>(`/projects/${enc(projectId)}/speakers/promote`, { method: "POST", json: body }),
};

export const voiceApi = {
  defaults: (signal?: AbortSignal) => request<DefaultVoiceList>("/default-voices", { signal }),
  engines: (signal?: AbortSignal) => request<EngineCatalog>("/engines", { signal }),
  list: (filters: {
    engine?: string;
    kind?: string;
    approved?: boolean;
    include_inactive?: boolean;
    include_legacy?: boolean;
    project_id?: string;
  } = {}, signal?: AbortSignal) => {
    const query = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined) query.set(key, String(value));
    });
    const queryString = query.toString();
    const suffix = queryString ? `?${queryString}` : "";
    return request<VoiceProfileList>(`/voice-library${suffix}`, { signal });
  },
  create: (body: {
    voice_name: string;
    engine: string;
    gender?: string;
    description?: string;
    voice_description_prompt?: string;
    sample_text?: string;
    reference_transcript?: string;
  }) => request<VoiceProfile>("/voice-library", { method: "POST", json: body }),
  update: (id: string, body: { active?: boolean; voice_description_prompt?: string; description?: string }) =>
    request<VoiceProfile>(`/voice-library/${enc(id)}`, { method: "PATCH", json: body }),
  uploadReference: (id: string, file: File, transcript: string) => {
    const form = new FormData();
    form.append("file", file);
    form.append("transcript", transcript);
    return upload<VoiceProfile>(`/voice-library/${enc(id)}/reference`, form).promise;
  },
  audition: (id: string, text: string) =>
    request<VoiceAudition>(`/voice-library/${enc(id)}/audition`, { method: "POST", json: { text } }),
  auditionStatus: (id: string, signal?: AbortSignal) =>
    request<VoiceAudition>(`/voice-library/auditions/${enc(id)}`, { signal }),
  approve: (id: string, approved: boolean, note = "") =>
    request<VoiceProfile>(`/voice-library/${enc(id)}/approval`, { method: "POST", json: { approved, note } }),
  runs: (id: string, signal?: AbortSignal) =>
    request<{ voice_id: string; runs: Record<string, unknown>[] }>(`/voice-library/${enc(id)}/runs`, { signal }),
};

export const characterApi = {
  list: (projectId: string, signal?: AbortSignal) =>
    request<SeriesCharacter[]>(`/projects/${enc(projectId)}/characters`, { signal }),
  create: (projectId: string, body: {
    canonical_name: string;
    aliases?: string[];
    description?: string;
    role?: string;
    gender?: string;
  }) => request<SeriesCharacter>(`/projects/${enc(projectId)}/characters`, { method: "POST", json: body }),
  addVoice: (projectId: string, characterId: string, body: {
    voice_source: "cast";
    voice_asset_id: string;
    label?: string;
    make_active: boolean;
    approved: boolean;
  }) => request<SeriesCharacter>(
    `/projects/${enc(projectId)}/characters/${enc(characterId)}/voice-profiles`,
    { method: "POST", json: body },
  ),
};

export const artifactApi = {
  timing: (jobId: string) => request<TimingReport>(`/jobs/${enc(jobId)}/timing-report`),
  resolution: (jobId: string) => request<Record<string, ResolvedVoice>>(`/jobs/${enc(jobId)}/voice-resolution`),
  report: (jobId: string) => request<JobReport>(`/jobs/${enc(jobId)}/report`),
  qa: (jobId: string, signal?: AbortSignal) => request<QaReport>(`/jobs/${enc(jobId)}/qa-report`, { signal }),
};

// Plain URLs for <video>/<audio>/<a download>; served with HTTP Range support by the backend.
export const urls = {
  sourceVideo: (jobId: string) => `${API_BASE}/jobs/${enc(jobId)}/source-video`,
  outputVideo: (jobId: string) => `${API_BASE}/jobs/${enc(jobId)}/output-video`,
  result: (jobId: string) => `${API_BASE}/jobs/${enc(jobId)}/result`,
  manifest: (jobId: string) => `${API_BASE}/jobs/${enc(jobId)}/manifest?download=true`,
  timing: (jobId: string) => `${API_BASE}/jobs/${enc(jobId)}/timing-report`,
  report: (jobId: string) => `${API_BASE}/jobs/${enc(jobId)}/report`,
  qaReport: (jobId: string) => `${API_BASE}/jobs/${enc(jobId)}/qa-report`,
  stem: (jobId: string, stem: string) => `${API_BASE}/jobs/${enc(jobId)}/stems/${enc(stem)}`,
  originalSegment: (jobId: string, segId: string) =>
    `${API_BASE}/jobs/${enc(jobId)}/segments/${enc(segId)}/original-audio`,
  dubSegment: (jobId: string, segId: string) => `${API_BASE}/jobs/${enc(jobId)}/segments/${enc(segId)}/dub-audio`,
  jobSpeakerReference: (jobId: string, spk: string) =>
    `${API_BASE}/jobs/${enc(jobId)}/speakers/${enc(spk)}/reference-audio`,
  libraryReference: (projectId: string, id: string, version?: string | null) =>
    `${API_BASE}/projects/${enc(projectId)}/speakers/${enc(id)}/reference${version ? `?v=${enc(version)}` : ""}`,
  defaultVoice: (id: string) => `${API_BASE}/default-voices/${enc(id)}/audio`,
  voicePreview: (id: string, runId?: string) =>
    `${API_BASE}/voice-library/${enc(id)}/preview${runId ? `?run_id=${enc(runId)}` : ""}`,
  voiceReference: (id: string) => `${API_BASE}/voice-library/${enc(id)}/reference`,
};
