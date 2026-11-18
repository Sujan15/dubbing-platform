# Hindi AI Dubbing Platform (Windows native, no Docker)

A project-based dubbing workstation: dubs English, Mandarin and Korean films, series and animation
into Hindi with a **human review step before any voice is generated**, a **per-project character
voice library** reused across episodes, and **segment-level timing** that keeps every line on the
original timeline. Lip-sync is out of scope.

```
CREATE PROJECT ─► UPLOAD VIDEO ─► ANALYSIS (automatic, no TTS)
   ffprobe ─► audio extraction ─► Demucs (vocals | background) ─► faster-whisper large-v3
   ─► pyannote 3.1 diarization ─► speaker-attributed lines ─► LLM translation → Hindi
   ─► clone-reference statistics per speaker ─► editable draft            ─► AWAITING_REVIEW
REVIEW (you): edit Hindi / transcript / timing, rename · merge · split speakers,
              choose an approved Parler profile or transcript-bearing IndicF5 reference
CONFIRM & GENERATE ─► immutable approved snapshot ─► voice resolution ─► Parler / IndicF5 per line
   ─► timing (fit / atempo / LLM shorten for machine lines / capped overflow) ─► mix ─► mux
   ─► output validation                                                   ─► SUCCEEDED
```

XTTS assignments and automatic fallback are retired. Project jobs use the review gate to select an approved
Parler profile or a transcript-bearing IndicF5 reference before synthesis.

## Layout

```
setup_and_run.bat                 venv + dependencies + web UI build + launch (Windows)
requirements.txt                  API, analysis and audio dependencies (no XTTS package)
.env.example                      every setting, documented
ruff.toml                         Python lint rules
backend/app/config.py             Pydantic settings (env / .env)
backend/app/main.py               FastAPI app, original single-shot routes, error mapping, serves the web UI
backend/app/api.py                project, video, draft review, speaker library, default voice, artifact routes
backend/app/jobs.py               job manager: states, review gate, confirm, cancel, retry, restart resume
backend/services/pipeline.py      DubbingPipeline: analyze / generate / run, stage cache, Parler / IndicF5
backend/services/drafts.py        editable draft manifest, patching, confirmation validation
backend/services/projects.py      projects, videos, drafts, speaker library (JSON + WAV storage)
backend/services/voices.py        reference validation, default voice bank, voice resolution
backend/services/storage.py       atomic JSON writes, id validation
backend/services/media.py         FFmpeg / FFprobe / WAV helpers
backend/services/translation.py   OpenRouter + Ollama clients
frontend/                         React + TypeScript + Vite web UI (built into frontend/dist)
tests/test_pipeline.py            original pipeline/API tests (unchanged)
tests/test_projects.py            project, library, review-gate, resume and reuse tests
```

## Prerequisites

| Requirement | How to get it |
|---|---|
| Windows 10/11 x64, Python 3.10–3.12 (3.11 recommended) | `winget install --id Python.Python.3.11 -e` |
| FFmpeg + FFprobe on PATH | `winget install --id Gyan.FFmpeg -e` (open a new terminal afterwards) |
| Node.js 20+ (only to build/develop the UI) | `winget install --id OpenJS.NodeJS.LTS -e` |
| NVIDIA GPU (optional, strongly recommended) | ≥ 8 GB VRAM; RTX 50-series gets the CUDA 12.8 build automatically |
| Hugging Face token | accept **pyannote/speaker-diarization-3.1** and **pyannote/segmentation-3.0** with the token's account |
| OpenRouter API key (or Ollama) | openrouter.ai/keys — or `ollama pull <model>` |
| TTS models | Parler and IndicF5 are loaded through their configured isolated worker environments |

## Windows setup and execution (PowerShell)

```powershell
cd "G:\ETL DEMO WORK\ETL_BEST_PRACTICES\ai_dubbing\hindi-dubbing-platform"

# first run: creates .env from .env.example and stops - fill in HF_TOKEN and OPENROUTER_API_KEY
.\setup_and_run.bat

# second run: installs dependencies (once), builds the web UI if needed, starts the server
.\setup_and_run.bat
```

* Web UI: <http://127.0.0.1:8000/> · API docs: <http://127.0.0.1:8000/docs>
* **Backend API and worker are one process.** Jobs run on a bounded worker thread pool inside the
  server (`MAX_CONCURRENT_JOBS`); HTTP handlers never run models or long FFmpeg work.
* CPU thread limits belong in the environment, not in an ad-hoc command line:

  ```powershell
  $env:OMP_NUM_THREADS = "4"; $env:MKL_NUM_THREADS = "4"; .\setup_and_run.bat
  ```

Run the server directly (after setup):

```powershell
.\.venv\Scripts\python.exe -m backend.app.main
```

Frontend development (hot reload on :5173, API proxied to :8000):

```powershell
cd frontend
npm ci
npm run dev
```

Rebuild the production UI after frontend changes: `cd frontend; npm run build`.

## Using the workstation

1. **Create a project** (Projects → New project). Target language is Hindi.
2. **Upload a video** (project → Upload video): drag-and-drop, source language (or auto-detect), mix mode,
   optional min/max speakers. Upload progress is real; analysis starts on the server afterwards.
3. **Analysis** runs extraction → separation → ASR → diarization → alignment → translation → clone
   reference statistics, then the job stops at **Awaiting review**. No TTS has run.
4. **Review workstation** (`Open review workstation`):
   * video player + speaker-lane timeline; click a line or a timeline block to seek;
     <kbd>Space</kbd> play/pause, <kbd>J</kbd>/<kbd>K</kbd> previous/next line, <kbd>Ctrl</kbd>+<kbd>S</kbd> save;
   * edit Hindi text (machine version kept, one-click revert), correct the transcript, adjust start/end,
     change a line's speaker, set review status, play the original line;
   * speaker rail: rename, merge into another speaker, add a new speaker (to split lines), filter lines,
     preview the reference voice, choose the **voice source** per character;
   * **Save draft** persists everything (survives refresh and server restarts). Saves are versioned:
     if the draft changed elsewhere you get a conflict prompt instead of a silent overwrite.
5. **Confirm & Generate** validates the draft (speaker assignments, voices, Devanagari text, timestamps),
   shows a summary with warnings, then locks an immutable `approved_v<N>.json` and starts generation.
   A second confirmation is rejected (409).
6. **Generation progress** shows the real stages and the current line count; no invented percentages.
   Cancel and retry are available; retry resumes from cached stages.
7. **Result**: dubbed video player, download, manifest, timing report, job report, voices actually used,
   timing warnings, and **Save to library** per speaker.

### Voice sources and resolution

| Source | Reference used | Notes |
|---|---|---|
| `clone` | up to `SPEAKER_REFERENCE_SECONDS` of the speaker's longest clean, non-overlapped lines in this video | needs `MIN_SPEAKER_REFERENCE_SECONDS` of clean speech to be considered reliable |
| `library` | `data/projects/<project>/speakers/<speaker>/reference.wav` | only speakers of the **same project**; validated audio and transcript required |
| `cast` | approved Voice Library profile | Parler prompt profile or IndicF5 reference profile |

Historical XTTS/default assignments remain readable but are not selectable or synthesized. Old project library
recordings without a transcript remain preserved and must have their transcript added before IndicF5 can use them.

Every decision (requested vs resolved source, engine, reference seconds) is written to
`voice_resolution.json` and `job_report.json` and shown in the UI. Nothing is substituted silently.

### Project Speaker Library

* Add a character, upload 15–20 s of one person speaking (WAV/FLAC/MP3/M4A/OGG). The server converts it to
  mono 16-bit PCM at `SPEAKER_REFERENCE_SAMPLE_RATE`, peak-normalises it and rejects files that are not
  decodable, silent, longer than 120 s or contain < 3 s of speech; < `MIN_SPEAKER_REFERENCE_SECONDS`
  of speech is accepted with a warning.
* Replacing a reference requires an explicit "replace" confirmation; the old file moves to `history/`.
* **Save to library** on a result (or `POST .../speakers/promote`) copies the voice a speaker used into the
  library (`cloned` or `default_promoted`).
* In episode 2+, set a detected speaker's voice source to *Project Speaker Library* and pick the character.
  Cross-video speaker identity is **your mapping** — automatic matching across videos is not attempted.
* A library speaker selected in an open review cannot be deleted.

### Recording a good reference

Mono or stereo, 16 kHz or higher, 15–20 s of continuous natural speech from one person, no music, no
other voices, no clipping, little room echo. Varied sentences beat one repeated phrase.

### Default Hindi voice bank

Put files in `data/default_voices` (or `DEFAULT_VOICES_DIR`): `<voice_id>.wav` where the id is
letters/digits/`_`/`-` (e.g. `woman_warm.wav`), plus an optional `<voice_id>.json`:

```json
{"label": "Woman – warm", "description": "Mid-30s, soft", "gender": "female"}
```

Files are validated on first use (decodable, ≥ 3 s of speech); invalid ones are listed and skipped. The
bank is read-only from the UI. An empty folder shows an empty state; nothing crashes.
The platform ships **no** voice files — use recordings you have the rights to.

## Output locations

| What | Where |
|---|---|
| Uploaded episode (immutable) | `data/projects/<project>/videos/<video>/original/input.<ext>` |
| Editable draft / approved snapshots | `.../videos/<video>/drafts/draft.json`, `approved_v<N>.json` |
| Library references | `data/projects/<project>/speakers/<speaker>/reference.wav` (+ `history/`) |
| Dubbed video, `manifest.json`, `timing_report.json`, `voice_resolution.json`, `job_report.json`, `analysis.json`, `job.log` | `data/outputs/<job_id>/` |
| Stage cache (content-addressed) | `data/work/<video-sha256>/<stage>/<key>/`, per-line TTS in `tts_cache/` |
| Job records | `data/jobs/<job_id>.json` |
| Single-shot uploads | `data/uploads/<job_id>/` |

## API

All routes are under `/api/v1`; interactive docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | FFmpeg version, configuration problems, job counts |
| GET | `/meta` | languages, mix modes, limits, review flag |
| POST | `/jobs` | **single-shot** upload → 202 job (unchanged) |
| GET | `/jobs/{id}` | job record: state, stage, `stage_history`, warnings, error |
| GET | `/jobs/{id}/result` | dubbed video download (unchanged) |
| GET | `/jobs` | list jobs (`?project_id=`) |
| POST | `/jobs/{id}/cancel` · `/jobs/{id}/retry` | cooperative cancel · retry from last phase |
| GET | `/jobs/{id}/logs?tail=200` | last lines of `job.log` |
| GET | `/jobs/{id}/draft` | editable draft |
| PATCH | `/jobs/{id}/draft` | save edits (`base_version` required; 409 on conflict) |
| GET | `/jobs/{id}/confirm-preview` | dry-run validation for the confirmation summary |
| POST | `/jobs/{id}/confirm` | lock approved snapshot, start generation (once) |
| GET | `/jobs/{id}/source-video` · `/output-video` | range-enabled streaming for the browser player |
| GET | `/jobs/{id}/segments/{seg}/original-audio` · `/dub-audio` | per-line previews |
| GET | `/jobs/{id}/speakers/{spk}/reference-audio` | clone reference preview |
| GET | `/jobs/{id}/manifest` · `/timing-report` · `/voice-resolution` · `/report` | artifacts |
| POST/GET | `/projects` | create / list (`?q=`) |
| GET/PATCH/DELETE | `/projects/{id}` | detail (videos + jobs + speakers) / update / delete (`?force=true` when not empty; refused while jobs run) |
| POST/GET | `/projects/{id}/videos` | upload (multipart) → `{project_id, video_id, job_id, status}` / list |
| GET | `/projects/{id}/videos/{video_id}` | video + current job |
| GET/POST | `/projects/{id}/speakers` | list / create |
| GET/PATCH/DELETE | `/projects/{id}/speakers/{sid}` | read / rename + notes / delete (dependency-checked) |
| POST/GET | `/projects/{id}/speakers/{sid}/reference` | upload (`replace=true` to overwrite) / stream |
| POST | `/projects/{id}/speakers/promote` | save a job speaker's voice into the library |
| GET | `/default-voices` · `/default-voices/{id}/audio` | valid default voices / preview |

Errors: `404` unknown id, `409` state or version conflict, `413` too large, `415` wrong type,
`422` validation (`{"detail": ..., "errors": [...]}`), `503` server configuration missing.

### Example (PowerShell)

```powershell
$api  = "http://127.0.0.1:8000/api/v1"
$json = "application/json; charset=utf-8"
function Send($method, $url, $body) {          # UTF-8 body so Devanagari survives Windows PowerShell 5.1
  Invoke-RestMethod -Method $method -Uri $url -ContentType $json `
    -Body ([Text.Encoding]::UTF8.GetBytes(($body | ConvertTo-Json -Depth 6)))
}

$p = Send POST "$api/projects" @{ name = "My Hindi Series"; target_language = "hi" }
$u = curl.exe -s -F "file=@D:\films\episode01.mkv" -F "source_language=ko" -F "mix_mode=preserve_background" `
       "$api/projects/$($p.project_id)/videos" | ConvertFrom-Json
do { Start-Sleep 10; $j = Invoke-RestMethod "$api/jobs/$($u.job_id)" } while ($j.state -in "queued","running")
$d = Invoke-RestMethod "$api/jobs/$($u.job_id)/draft"
$d = Send PATCH "$api/jobs/$($u.job_id)/draft" @{
  base_version = $d.version
  speakers = @(@{ speaker_id = "SPEAKER_00"; display_name = "Rahul"; voice_source = "default"; default_voice_id = "man_deep" })
  segments = @(@{ segment_id = "seg_00001"; translated_text = "नमस्ते, आप कैसे हैं?" })
}
Send POST "$api/jobs/$($u.job_id)/confirm" @{ base_version = $d.version }
do { Start-Sleep 10; $j = Invoke-RestMethod "$api/jobs/$($u.job_id)" } while ($j.state -eq "generating")
curl.exe -o episode01.hindi.mkv "$api/jobs/$($u.job_id)/result"
```

The same sequence was executed end-to-end with `curl` against a running server (fake models) during
development; the PowerShell wrapper itself has not been run on Windows.

Draft response (abridged):

```json
{"version": 2, "review_status": "draft", "source_language": "ko", "media_duration": 2580.4,
 "segments": [{"segment_id": "seg_00001", "start": 0.5, "end": 2.8, "original_text": "...",
               "translated_text": "नमस्ते, आप कैसे हैं?", "machine_translation": "...", "speaker_id": "SPEAKER_00",
               "speaker_display_name": "Rahul", "confidence": 0.91, "review_status": "edited",
               "voice_source": "default", "default_voice_id": "man_deep", "warnings": []}],
 "speakers": [{"speaker_id": "SPEAKER_00", "display_name": "Rahul", "origin_labels": ["SPEAKER_00"],
               "segment_count": 212, "total_speech_duration": 640.2, "usable_reference_duration": 182.5,
               "voice_source": "default", "default_voice_id": "man_deep",
               "available_voice_options": ["clone", "library", "default"], "warnings": []}]}
```

## Job states

```
single-shot:  queued → running → succeeded | failed
project:      queued → running (analysis) → awaiting_review ──confirm──► generating → succeeded | failed
              cancel: from queued / running / awaiting_review / generating → cancelled
              retry:  failed | cancelled → queued (analysis) or generating (if a snapshot was approved)
restart:      queued/running are re-queued, generating resumes, awaiting_review keeps its draft
```

## Caching, invalidation and storage growth

Each stage's key hashes its version, parameters and input identities. Consequences:

* Confirming a reviewed draft re-uses extraction, separation, ASR, diarization and translation (tests
  assert these are not re-run).
* Clone references depend only on line timings and speaker assignment — not on text.
* The synthesis stage key includes every line's text, speaker and each speaker's reference **content
  hash**; per-line TTS clips are cached by text + reference hash + engine settings. Editing one line
  re-synthesizes that line only; replacing a library reference re-synthesizes that character's lines.
* Two uploads of byte-identical video share the analysis cache, but each video keeps its own draft.

Storage grows with `data/work` (≈ 10× the audio size per video plus TTS clips) and `data/outputs`. Delete
`data/work/<hash>` to drop one video's cache, or a job's `data/outputs/<job_id>` after downloading.
Deleting a project removes its uploads, drafts, library and job outputs, not the shared `data/work` cache.

## Configuration added in this version

| Variable | Default | Meaning |
|---|---|---|
| `PROJECT_REVIEW_ENABLED` | `true` | project uploads stop at `awaiting_review` (per-project override in project settings) |
| `DEFAULT_VOICES_DIR` | `data/default_voices` | historical read-only preview folder; not used for synthesis |
| `MIN_SPEAKER_REFERENCE_SECONDS` | `6` | clean speech needed for a reliable clone |
| `SPEAKER_REFERENCE_SAMPLE_RATE` | `22050` | library reference format (16000 / 22050 / 24000) |
| `MAX_REFERENCE_UPLOAD_MB` | `50` | reference upload limit |
| `TIMING_MAX_SPEEDUP` | alias | alias of the existing `TIMING_MAX_ATEMPO` |
| `TIMING_MAX_TTS_SPEED` | `1.0` | native speaking-rate cap tried before atempo (1.0 = off) |
| `TIMING_MAX_OVERFLOW_SECONDS` | `0.0` | allowed run-on into following silence (recommended 0.3–0.5) |

Existing variables are unchanged; see `.env.example`.

## Tests and checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip install ruff
.\.venv\Scripts\ruff.exe check backend tests
cd frontend; npm ci; npm run typecheck; npm run lint; npm test; npm run build
```

Python tests need FFmpeg only (models are deterministic fakes); frontend tests use a mocked `fetch`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `503 configuration_problems` on upload | set `HF_TOKEN` and `OPENROUTER_API_KEY` (or `TRANSLATION_ENGINE=ollama`) |
| Web UI shows 404 at `/` | `frontend\dist` missing: install Node.js and run `setup_and_run.bat` again, or `cd frontend; npm ci; npm run build` |
| "preserve_background needs separated stems" warning | no CUDA GPU: set `SEPARATION_ALLOW_CPU=true` (slow) or accept `original_plus_dub` |
| IndicF5 reference cannot be selected | add the exact words spoken to the project library reference transcript, or select an approved Parler profile |
| `409 Draft was modified` | someone saved in another tab; reload the draft |
| Default voice not listed | check the `invalid` list in `GET /api/v1/default-voices` (silent / too short / unreadable) |
| `OMP: Error #15` | `KMP_DUPLICATE_LIB_OK=TRUE` is set automatically; do not unset it |

## Known limitations

* **CPU**: correct but slow — Whisper large-v3 and Hindi speech synthesis can take many hours for a feature film.
* **Voice cloning**: approximate similarity only; it degrades sharply with < 6 s of clean speech, music
  under the voice, or overlapping speakers. Never claim identity-level cloning.
* **Diarization**: pyannote may split one actor into two labels or merge similar voices; fix it with
  merge / reassign in review. Cross-episode identity is user-mapped via the library.
* **Background separation**: Demucs leaves artefacts and some original dialogue bleed; without a GPU it is
  skipped unless `SEPARATION_ALLOW_CPU=true`.
* **Timing**: human-edited lines are never rewritten by the LLM; lines that still do not fit after the
  speed-up cap are flagged `atempo_capped_overflow` in the timing report, never truncated or dropped.
* **Languages**: only Hindi is a validated target; source tuning covers en / zh / ko.
* **Security**: no authentication; the server binds to `127.0.0.1`. Put it behind an authenticating
  reverse proxy before exposing it on a network. Project boundaries are enforced per request (library
  voices of another project cannot be selected), and the data model has a single owner today.

## Version notes

* `huggingface-hub<1.0` is retained for pyannote 3.x authentication.
* pyannote 3.x checkpoints need `torch.load(weights_only=False)`; enabled only while loading diarization.
* `KMP_DUPLICATE_LIB_OK=TRUE` because torch and ctranslate2 both ship `libiomp5md.dll` on Windows.


# DEPLOY IN BASH BELOW COMMAND
source .venv/Scripts/activate
cmd //c "set OMP_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& setup_and_run.bat"
