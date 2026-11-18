// Mirrors the backend Pydantic schemas (backend/app/jobs.py, backend/services/{projects,drafts}.py).

export type JobState =
  | "queued"
  | "running"
  | "awaiting_review"
  | "generating"
  | "succeeded"
  | "failed"
  | "cancelled";

export type MixMode = "preserve_background" | "dub_only" | "original_plus_dub";
export type VoiceSource = "cast" | "character" | "clone" | "library" | "default";
export type SegmentReviewStatus = "pending" | "edited" | "approved";

export interface JobOptions {
  source_language: string | null;
  mix_mode: MixMode;
  translation_engine: string;
  min_speakers: number | null;
  max_speakers: number | null;
}

export interface StageEvent {
  stage: string;
  phase: "analysis" | "generation";
  status: "running" | "completed" | "failed" | "cancelled";
  started_at: string;
  finished_at: string | null;
  message: string;
}

export interface JobRecord {
  job_id: string;
  state: JobState;
  stage: string;
  progress: number;
  message: string;
  input_filename: string;
  options: JobOptions;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  source_language: string | null;
  speakers: number | null;
  segments: number | null;
  warnings: string[];
  error: string | null;
  project_id: string | null;
  video_id: string | null;
  review_required: boolean;
  phase: "analysis" | "generation";
  approved_version: number | null;
  stage_history: StageEvent[];
  analysis_completed_at: string | null;
  generation_started_at: string | null;
  qa_status: string | null;
  qa_report_path: string | null;
  release_approved: boolean;
  release_note: string;
  engines: string[];
}

export interface ProjectSettings {
  source_language: string | null;
  mix_mode: MixMode;
  translation_engine: string | null;
  review_enabled: boolean | null;
}

export interface Project {
  project_id: string;
  name: string;
  description: string;
  target_language: string;
  settings: ProjectSettings;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface ProjectSummary extends Project {
  video_count: number;
  speaker_count: number;
  jobs: Record<JobState, number>;
  last_activity_at: string | null;
}

export interface ProjectVideo {
  video_id: string;
  project_id: string;
  title: string;
  original_filename: string;
  size_bytes: number;
  duration: number;
  width: number;
  height: number;
  fps: number;
  job_ids: string[];
  current_job_id: string | null;
  created_at: string;
  updated_at: string;
  job: JobRecord | null;
}

export interface LibrarySpeaker {
  speaker_id: string;
  project_id: string;
  display_name: string;
  source: "cloned" | "uploaded" | "default_promoted" | null;
  reference_available: boolean;
  reference_duration_seconds: number;
  reference_speech_seconds: number;
  reference_sample_rate: number | null;
  reference_transcript: string | null;
  reference_updated_at: string | null;
  notes: string;
  origin: Record<string, string>;
  warnings: string[];
  created_at: string;
  updated_at: string;
}

export interface ProjectDetail extends ProjectSummary {
  videos: ProjectVideo[];
  speakers: LibrarySpeaker[];
}

export interface VideoUploadResponse {
  project_id: string;
  video_id: string;
  job_id: string;
  status: JobState;
  video: ProjectVideo;
}

export interface DraftSegment {
  segment_id: string;
  start: number;
  end: number;
  original_text: string;
  translated_text: string;
  machine_translation: string;
  speaker_id: string;
  confidence: number | null;
  review_status: SegmentReviewStatus;
  warnings: string[];
  speaker_display_name: string | null;
  voice_source: VoiceSource | null;
  default_voice_id: string | null;
  library_speaker_id: string | null;
  reference_duration_seconds: number | null;
  character_id?: string | null;
  character_name?: string | null;
}

export interface DraftSpeaker {
  speaker_id: string;
  display_name: string;
  color: number;
  origin_labels: string[];
  segment_count: number;
  total_speech_duration: number;
  usable_reference_duration: number;
  reference_available: boolean;
  voice_source: VoiceSource;
  library_speaker_id: string | null;
  default_voice_id: string | null;
  available_voice_options: VoiceSource[];
  warnings: string[];
  engine?: string;
  voice_id?: string | null;
  character_id?: string | null;
  character_name?: string | null;
  mapping_status?: string;
  suggested_character_id?: string | null;
  suggested_character_name?: string | null;
  identity_confidence?: number | null;
  identity_reason?: string;
  identity_candidates?: { character_id: string; character_name?: string; confidence?: number; reason?: string }[];
}

export interface DraftManifest {
  manifest_version: number;
  version: number;
  job_id: string;
  project_id: string;
  video_id: string;
  source_language: string;
  target_language: string;
  pipeline_version: string;
  media_duration: number;
  mix_mode: MixMode;
  created_at: string;
  updated_at: string;
  review_status: "draft" | "approved";
  approved_version: number | null;
  segments: DraftSegment[];
  speakers: DraftSpeaker[];
  warnings: string[];
}

export interface SegmentPatch {
  segment_id: string;
  translated_text?: string;
  original_text?: string;
  speaker_id?: string;
  start?: number;
  end?: number;
  review_status?: SegmentReviewStatus;
}

export interface SpeakerPatch {
  speaker_id: string;
  display_name?: string;
  voice_source?: VoiceSource;
  library_speaker_id?: string | null;
  default_voice_id?: string | null;
  engine?: string;
  voice_id?: string | null;
  character_id?: string | null;
  reject_suggestion?: boolean;
}

export interface DraftPatch {
  base_version: number;
  new_speakers?: { speaker_id: string; display_name: string }[];
  merges?: { source_speaker_id: string; target_speaker_id: string }[];
  speakers?: SpeakerPatch[];
  segments?: SegmentPatch[];
  mix_mode?: MixMode;
}

export interface ConfirmPreview {
  version: number;
  errors: string[];
  warnings: string[];
  speakers: {
    speaker_id: string;
    display_name: string;
    voice_source: VoiceSource;
    library_speaker_id: string | null;
    default_voice_id: string | null;
    engine?: string;
    voice_id?: string | null;
    character_id?: string | null;
    character_name?: string | null;
    segment_count: number;
  }[];
}

export interface DefaultVoice {
  voice_id: string;
  label: string;
  description: string;
  gender: string | null;
  duration_seconds: number;
  speech_seconds: number;
  sample_rate: number;
  valid: boolean;
  problem: string | null;
}

export interface DefaultVoiceList {
  voices: DefaultVoice[];
  invalid: { voice_id: string; problem: string }[];
}

export interface Meta {
  source_languages: Record<string, string>;
  target_languages: Record<string, string>;
  default_target_language: string;
  mix_modes: MixMode[];
  default_mix_mode: MixMode;
  translation_engines: string[];
  default_translation_engine: string;
  review_enabled: boolean;
  min_speaker_reference_seconds: number;
  max_upload_mb: number;
  max_reference_upload_mb: number;
  video_extensions: string[];
  reference_extensions: string[];
  configuration_problems: string[];
}

export interface ResolvedVoice {
  speaker_id: string;
  display_name: string;
  requested_source: VoiceSource;
  resolved_source: VoiceSource;
  reference_seconds: number;
  library_speaker_id: string | null;
  default_voice_id: string | null;
  fallback_applied: boolean;
  warnings: string[];
}

export interface TimingSegment {
  segment_id: string;
  speaker_id: string;
  status: string;
  original_duration: number;
  window_seconds: number;
  generated_duration: number;
  final_duration: number;
  speed_adjustment: number;
  tts_speed: number;
  placement_start: number;
  placement_end: number;
  overflow_seconds: number;
  overlaps_next: boolean;
  warning: string | null;
}

export interface TimingReport {
  summary: Record<string, number>;
  segments: TimingSegment[];
}

export interface JobReport {
  source_language: string;
  target_language: string;
  mix_mode_requested: MixMode;
  mix_mode_used: MixMode;
  separation: boolean;
  speakers: number;
  segments: number;
  voices: Record<string, ResolvedVoice>;
  warnings: string[];
  runtime: Record<string, string | null>;
}

export interface VoiceEngineInfo {
  engine: string;
  enabled: boolean;
  available: boolean;
  reason: string;
  model_id: string | null;
  model_revision: string | null;
  capabilities: {
    reference_conditioned: boolean;
    needs_reference_transcript: boolean;
    prompt_conditioned: boolean;
    supports_seed: boolean;
    supports_speed: boolean;
    languages: string[];
  };
  license: string;
  commercial_use: boolean | null;
  legacy: boolean;
}

export interface EngineCatalog {
  engines: VoiceEngineInfo[];
  enabled: string[];
  default_engine: string;
  default_clone_engine: string;
}

export interface VoiceProfile {
  voice_id: string;
  voice_name: string;
  engine: string;
  kind: "cast" | "clone" | "legacy";
  gender: string | null;
  description: string;
  voice_description_prompt: string | null;
  prompt_version: number;
  prompt_versions: { version: number; prompt: string; created_at: string }[];
  sample_text: string | null;
  active: boolean;
  approved: boolean;
  preview: { duration_seconds: number; generated_at: string; run_id: string; problems: string[] } | null;
  preview_url: string | null;
  selectable?: boolean;
  needs_reaudition?: boolean;
  reference_available: boolean;
  reference_transcript: string | null;
  created_at: string;
  updated_at: string;
  legacy?: boolean;
  voice_source?: VoiceSource;
}

export interface VoiceProfileList {
  voices: VoiceProfile[];
  engines: VoiceEngineInfo[];
}

export interface VoiceAudition {
  audition_id: string;
  voice_id: string;
  text: string;
  status: "queued" | "running" | "done" | "failed";
  problems?: string[];
  error: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface CharacterVoiceProfile {
  voice_profile_id: string;
  voice_source: VoiceSource;
  engine: string;
  voice_asset_id: string | null;
  label: string;
  notes: string;
  approved: boolean;
  identity_locked: boolean;
}

export interface SeriesCharacter {
  character_id: string;
  series_id: string;
  canonical_name: string;
  aliases: string[];
  description: string;
  role: string | null;
  gender: string | null;
  status: string;
  voice_profiles: CharacterVoiceProfile[];
  active_voice_profile_id: string | null;
  has_approved_voice: boolean;
  episodes: Record<string, unknown>[];
  embedding_count: number;
}

export interface QaCheck {
  status: string;
  value: unknown;
  detail: string;
}

export interface QaReport {
  status: string;
  checks: Record<string, QaCheck>;
  metrics: Record<string, unknown>;
  clipping: Record<string, unknown>;
  release_blocked: boolean;
  note: string;
}
