"""Project workflow tests: projects, Speaker Library, default voices, voice resolution, the review
gate (no TTS before confirmation), draft editing/concurrency, confirmation, resume from cache and
library reuse across a second video. Models are the same deterministic fakes as test_pipeline.py.

    .venv\\Scripts\\python -m pytest -q
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import test_pipeline as tp
from fastapi.testclient import TestClient

from backend.app import config, main
from backend.app.config import Settings, TTSEngine
from backend.services import pipeline as pl
from backend.services import tts
from backend.services.voices import (
    DefaultVoiceBank,
    VoiceRequest,
    VoiceResolutionError,
    VoiceSource,
    resolve_voice,
)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


# ------------------------------------------------------------------------------- fixtures
def _tone(path: Path, seconds: float, amp: float = 0.3, sr: int = 22050, freq: float = 180.0) -> Path:
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr, subtype="PCM_16")
    return path


@pytest.fixture()
def video(tmp_path: Path) -> Path:
    out = tmp_path / "episode 01.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=7",
                    "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=7",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
    return out


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(_env_file=None, data_dir=tmp_path / "data", separation_enabled=False, device="cpu",
                 openrouter_api_key="test-key", hf_token="hf_test",
                 tts_engines_enabled="parler,indicf5", timing_max_refine_attempts=2,
                 min_speaker_reference_seconds=2.0)
    s.ensure_dirs()
    _tone(s.default_voices_dir / "narrator.wav", 6.0)
    (s.default_voices_dir / "narrator.json").write_text(json.dumps({"label": "Narrator", "gender": "male"}))
    _tone(s.default_voices_dir / "broken.wav", 1.0, amp=0.0)  # silent -> listed as invalid only
    return s


@pytest.fixture()
def fake_models(monkeypatch):
    calls = {"transcribe": 0, "diarize": 0, "translate": 0}

    def transcribe(self, audio, out, language, duration, p):
        calls["transcribe"] += 1
        segs = [{"start": s, "end": e, "text": t, "avg_logprob": -0.1, "no_speech_prob": 0.01,
                 "words": [{"start": s, "end": e, "word": " " + t, "probability": 0.9}]} for s, e, _, t in tp.DIALOGUE]
        pl._write_json(out / "transcript.json", {"language": "en", "language_probability": 0.99, "segments": segs})

    def diarize(self, audio, out, options, p):
        calls["diarize"] += 1
        pl._write_json(out / "diarization.json",
                       [{"start": s, "end": e, "speaker": spk} for s, e, spk, _ in tp.DIALOGUE])

    class CountingTranslator(tp.FakeTranslator):
        def translate(self, lines, source_language, progress=None):
            calls["translate"] += 1
            return super().translate(lines, source_language, progress)

    monkeypatch.setattr(pl.DubbingPipeline, "_transcribe", transcribe)
    monkeypatch.setattr(pl.DubbingPipeline, "_diarize", diarize)
    monkeypatch.setattr(pl, "Translator", CountingTranslator)
    monkeypatch.setitem(tts._REGISTRY, TTSEngine.PARLER, tp.FakeParler)
    monkeypatch.setitem(tts._REGISTRY, TTSEngine.INDICF5, tp.FakeIndicF5)
    tp.FakeTranslator.shorten_calls = []
    tp.FakeParler.calls = 0
    tp.FakeIndicF5.calls = 0
    return calls


@pytest.fixture()
def client(settings, monkeypatch):
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    with TestClient(main.app) as c:
        yield c


def _wait(client, job_id: str, states: set[str], timeout: float = 60) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["state"] in states:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} stuck in {job['state']}: {job}")


def _upload(client, project_id: str, video: Path, **form) -> dict:
    with open(video, "rb") as f:
        r = client.post(f"/api/v1/projects/{project_id}/videos", files={"file": (video.name, f, "video/mp4")},
                        data={"source_language": "en", "mix_mode": "dub_only", **form})
    assert r.status_code == 202, r.text
    return r.json()


# ------------------------------------------------------------------------------- projects
def test_project_crud_and_delete_rules(client):
    r = client.post("/api/v1/projects", json={"name": "My Hindi Series", "description": "S1"})
    assert r.status_code == 201, r.text
    p = r.json()
    assert p["target_language"] == "hi" and p["video_count"] == 0
    assert client.post("/api/v1/projects", json={"name": "my hindi series"}).status_code == 409
    assert client.post("/api/v1/projects", json={"name": "X", "target_language": "fr"}).status_code == 422
    assert [x["name"] for x in client.get("/api/v1/projects", params={"q": "hindi"}).json()] == ["My Hindi Series"]

    pid = p["project_id"]
    r = client.patch(f"/api/v1/projects/{pid}", json={"name": "Renamed", "settings": {"mix_mode": "dub_only"}})
    assert r.json()["name"] == "Renamed" and r.json()["settings"]["mix_mode"] == "dub_only"
    assert client.get("/api/v1/projects/not-an-id").status_code == 404
    assert client.get("/api/v1/projects/../../etc").status_code == 404

    client.post(f"/api/v1/projects/{pid}/speakers", json={"display_name": "Rahul"})
    assert client.delete(f"/api/v1/projects/{pid}").status_code == 409  # has content, no force
    assert client.delete(f"/api/v1/projects/{pid}", params={"force": True}).status_code == 204
    assert client.get(f"/api/v1/projects/{pid}").status_code == 404


def test_video_upload_validation(client, video):
    pid = client.post("/api/v1/projects", json={"name": "P"}).json()["project_id"]
    r = client.post(f"/api/v1/projects/{pid}/videos", files={"file": ("a.txt", b"x", "text/plain")})
    assert r.status_code == 415
    r = client.post(f"/api/v1/projects/{pid}/videos", files={"file": ("a.mp4", b"not video", "video/mp4")})
    assert r.status_code == 422
    assert client.get(f"/api/v1/projects/{pid}/videos").json() == []  # failed uploads leave nothing behind


# ------------------------------------------------------------------------------- speaker library
def test_speaker_library_crud_and_reference_validation(client, tmp_path):
    pid = client.post("/api/v1/projects", json={"name": "Lib"}).json()["project_id"]
    spk = client.post(f"/api/v1/projects/{pid}/speakers", json={"display_name": "Rahul", "notes": "lead"}).json()
    sid = spk["speaker_id"]
    assert spk["reference_available"] is False
    assert client.post(f"/api/v1/projects/{pid}/speakers", json={"display_name": "rahul"}).status_code == 409

    def up(path: Path, replace=False):
        with open(path, "rb") as f:
            return client.post(f"/api/v1/projects/{pid}/speakers/{sid}/reference",
                               files={"file": (path.name, f, "audio/wav")}, data={"replace": str(replace).lower()})

    assert up(_tone(tmp_path / "silent.wav", 5, amp=0.0)).status_code == 422
    assert up(_tone(tmp_path / "short.wav", 1.5)).status_code == 422
    garbage = tmp_path / "garbage.wav"
    garbage.write_bytes(b"RIFF....not audio")
    assert up(garbage).status_code == 422
    bad_ext = tmp_path / "x.exe"
    bad_ext.write_bytes(b"MZ")
    assert up(bad_ext).status_code == 415

    r = up(_tone(tmp_path / "rahul.wav", 8.0, sr=44100))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reference_available"] and body["source"] == "uploaded" and body["reference_sample_rate"] == 22050
    assert 7.5 < body["reference_duration_seconds"] < 8.5
    info = sf.info(r"%s" % (Path(client.app.state.settings.projects_dir) / pid / "speakers" / sid / "reference.wav"))
    assert info.channels == 1 and info.samplerate == 22050 and info.subtype == "PCM_16"

    assert up(_tone(tmp_path / "rahul2.wav", 9.0)).status_code == 409  # no silent overwrite
    assert up(_tone(tmp_path / "rahul2.wav", 9.0), replace=True).status_code == 200
    hist = Path(client.app.state.settings.projects_dir) / pid / "speakers" / sid / "history"
    assert len(list(hist.glob("*.wav"))) == 1

    r = client.get(f"/api/v1/projects/{pid}/speakers/{sid}/reference", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    r = client.patch(f"/api/v1/projects/{pid}/speakers/{sid}", json={"display_name": "Rahul K", "notes": "n"})
    assert r.json()["display_name"] == "Rahul K"
    assert client.delete(f"/api/v1/projects/{pid}/speakers/{sid}").status_code == 204
    assert client.get(f"/api/v1/projects/{pid}/speakers/{sid}").status_code == 404


def test_default_voice_discovery(client, settings):
    body = client.get("/api/v1/default-voices").json()
    assert [v["voice_id"] for v in body["voices"]] == ["narrator"]
    assert body["voices"][0]["label"] == "Narrator" and "path" not in body["voices"][0]
    assert body["invalid"][0]["voice_id"] == "broken"
    assert client.get("/api/v1/default-voices/narrator/audio").status_code == 200
    assert client.get("/api/v1/default-voices/..%2Fx/audio").status_code == 404
    shutil.rmtree(settings.default_voices_dir)
    assert client.get("/api/v1/default-voices").json()["voices"] == []  # empty state, no crash


# ------------------------------------------------------------------------------- voice resolution
def test_voice_resolution_rules(settings, tmp_path):
    bank = DefaultVoiceBank(settings)
    clone = _tone(tmp_path / "clone.wav", 3.0)
    lib = _tone(tmp_path / "lib.wav", 5.0)
    lookup = {"a" * 32: (lib, 5.0, "Rahul", "नमस्ते दुनिया")}.get

    rv = resolve_voice(VoiceRequest("S0", "Rahul", VoiceSource.LIBRARY, library_speaker_id="a" * 32),
                       clone, 3.0, lookup, bank, settings)
    assert rv.resolved_source == VoiceSource.LIBRARY and rv.reference_path == lib
    assert rv.engine == TTSEngine.INDICF5 and rv.spec.reference_transcript == "नमस्ते दुनिया"
    with pytest.raises(VoiceResolutionError):  # other project's / unknown id
        resolve_voice(VoiceRequest("S0", "X", VoiceSource.LIBRARY, library_speaker_id="b" * 32),
                      clone, 3.0, lookup, bank, settings)
    with pytest.raises(VoiceResolutionError):
        resolve_voice(VoiceRequest("S0", "N", VoiceSource.DEFAULT, default_voice_id="narrator"),
                      None, 0, None, bank, settings)
    with pytest.raises(VoiceResolutionError):
        resolve_voice(VoiceRequest("S0", "N", VoiceSource.DEFAULT, default_voice_id="broken"), None, 0, None, bank, settings)

    req = VoiceRequest("S0", "C", engine=TTSEngine.INDICF5)
    rv = resolve_voice(req, clone, 3.0, None, bank, settings, clone_transcript="नमस्ते दुनिया")
    assert rv.resolved_source == VoiceSource.CLONE and not rv.warnings
    rv = resolve_voice(req, clone, 1.0, None, bank, settings, clone_transcript="नमस्ते दुनिया")
    assert rv.resolved_source == VoiceSource.CLONE and rv.warnings
    fb = settings.model_copy(update={"default_voice_fallback": "narrator"})
    with pytest.raises(VoiceResolutionError):
        resolve_voice(req, None, 0.0, None, DefaultVoiceBank(fb), fb, clone_transcript="नमस्ते दुनिया")
    with pytest.raises(VoiceResolutionError):
        resolve_voice(VoiceRequest("S0", "C", engine=TTSEngine.INDICF5), None, 0.0, None, bank, settings,
                      clone_transcript="नमस्ते दुनिया")


# ------------------------------------------------------------------------------- review workflow
def test_review_gate_draft_confirm_resume_and_library_reuse(client, settings, video, fake_models, tmp_path):
    pid = client.post("/api/v1/projects", json={"name": "Series"}).json()["project_id"]
    up = _upload(client, pid, video, title="Episode 01")
    job_id = up["job_id"]
    job = _wait(client, job_id, {"awaiting_review", "failed"})
    assert job["state"] == "awaiting_review", job
    assert tp.FakeParler.calls == tp.FakeIndicF5.calls == 0, "no TTS before confirmation"
    assert not list((settings.work_dir).rglob("synthesize"))
    assert {e["stage"] for e in job["stage_history"]} >= {"transcribe", "diarize", "translate", "voices"}

    draft = client.get(f"/api/v1/jobs/{job_id}/draft").json()
    assert draft["version"] == 1 and draft["review_status"] == "draft" and len(draft["segments"]) == 3
    assert [s["speaker_id"] for s in draft["speakers"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert draft["segments"][0]["speaker_display_name"] == "Speaker 1" and draft["segments"][0]["confidence"] == 0.9
    sp0 = draft["speakers"][0]
    assert sp0["usable_reference_duration"] > 0 and "clone" in sp0["available_voice_options"]
    assert client.get(f"/api/v1/jobs/{job_id}/speakers/SPEAKER_00/reference-audio").status_code == 200
    assert client.get(f"/api/v1/jobs/{job_id}/segments/seg_00002/original-audio").headers["content-type"] == "audio/wav"
    r = client.get(f"/api/v1/jobs/{job_id}/source-video", headers={"Range": "bytes=10-"})
    assert r.status_code == 206 and r.headers["content-range"].startswith("bytes 10-")

    # edits: translation, rename, new speaker + reassignment, then merge it back
    patch = {"base_version": 1,
             "segments": [{"segment_id": "seg_00003", "translated_text": "भागो जल्दी"}],
             "speakers": [{"speaker_id": "SPEAKER_00", "display_name": "Rahul",
                           "voice_source": "clone", "engine": "indicf5"},
                          {"speaker_id": "SPEAKER_01", "display_name": "Priya",
                           "voice_source": "clone", "engine": "indicf5"}]}
    r = client.patch(f"/api/v1/jobs/{job_id}/draft", json=patch)
    assert r.status_code == 200, r.text
    d2 = r.json()
    assert d2["version"] == 2
    seg3 = next(s for s in d2["segments"] if s["segment_id"] == "seg_00003")
    assert seg3["review_status"] == "edited" and seg3["speaker_display_name"] == "Rahul"
    assert client.patch(f"/api/v1/jobs/{job_id}/draft", json=patch).status_code == 409  # stale base_version

    r = client.patch(f"/api/v1/jobs/{job_id}/draft", json={
        "base_version": 2, "new_speakers": [{"speaker_id": "NEW_1", "display_name": "Narrator"}],
        "segments": [{"segment_id": "seg_00001", "speaker_id": "NEW_1"}]})
    assert r.status_code == 200 and r.json()["version"] == 3
    r = client.patch(f"/api/v1/jobs/{job_id}/draft", json={
        "base_version": 3, "merges": [{"source_speaker_id": "NEW_1", "target_speaker_id": "SPEAKER_00"}]})
    d4 = r.json()
    assert [s["speaker_id"] for s in d4["speakers"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert d4["segments"][0]["speaker_id"] == "SPEAKER_00"

    bad = client.patch(f"/api/v1/jobs/{job_id}/draft", json={
        "base_version": 4, "segments": [{"segment_id": "seg_00001", "start": 2.0, "end": 2.05}]})
    assert bad.status_code == 422 and bad.json()["errors"]
    r = client.patch(f"/api/v1/jobs/{job_id}/draft", json={
        "base_version": 4, "speakers": [{"speaker_id": "SPEAKER_00", "voice_source": "library",
                                         "library_speaker_id": "f" * 32}]})
    assert r.status_code == 422  # not a speaker of this project

    # an empty translation blocks confirmation
    client.patch(f"/api/v1/jobs/{job_id}/draft",
                 json={"base_version": 4, "segments": [{"segment_id": "seg_00002", "translated_text": ""}]})
    preview = client.get(f"/api/v1/jobs/{job_id}/confirm-preview").json()
    assert preview["errors"]
    assert client.post(f"/api/v1/jobs/{job_id}/confirm", json={"base_version": 5}).status_code == 422
    client.patch(f"/api/v1/jobs/{job_id}/draft",
                 json={"base_version": 5, "segments": [{"segment_id": "seg_00002", "translated_text": "मैं ठीक हूँ"}]})

    # draft survives an API restart
    client.app.state.jobs.shutdown()
    from backend.app.jobs import JobManager
    client.app.state.jobs = JobManager(settings, client.app.state.projects)
    assert client.get(f"/api/v1/jobs/{job_id}").json()["state"] == "awaiting_review"
    assert client.get(f"/api/v1/jobs/{job_id}/draft").json()["version"] == 6

    asr_calls = dict(fake_models)
    r = client.post(f"/api/v1/jobs/{job_id}/confirm", json={"base_version": 6})
    assert r.status_code == 202, r.text
    assert r.json()["approved_version"] == 6
    assert client.post(f"/api/v1/jobs/{job_id}/confirm", json={"base_version": 6}).status_code == 409
    assert client.patch(f"/api/v1/jobs/{job_id}/draft", json={"base_version": 6}).status_code == 409
    job = _wait(client, job_id, {"succeeded", "failed"})
    assert job["state"] == "succeeded", job
    assert fake_models == asr_calls, "confirmation must not re-run ASR, diarization or translation"
    snap = settings.projects_dir / pid / "videos" / up["video_id"] / "drafts" / "approved_v6.json"
    assert snap.is_file()

    res = client.get(f"/api/v1/jobs/{job_id}/voice-resolution").json()
    assert res["SPEAKER_01"]["resolved_source"] == "clone" and res["SPEAKER_00"]["resolved_source"] == "clone"
    manifest = json.loads(Path(job["manifest_path"]).read_text(encoding="utf-8"))
    assert [m["speaker_display_name"] for m in manifest] == ["Rahul", "Priya", "Rahul"]
    assert manifest[2]["translated_text"] == "भागो जल्दी"  # human text spoken verbatim (locked from LLM refine)
    timing = client.get(f"/api/v1/jobs/{job_id}/timing-report").json()["segments"]
    assert [round(t["placement_start"], 2) for t in timing] == [0.5, 3.0, 4.5]  # original timeline kept
    assert all(t["status"] for t in timing)
    r = client.get(f"/api/v1/jobs/{job_id}/output-video", headers={"Range": "bytes=0-1023"})
    assert r.status_code == 206
    assert client.get(f"/api/v1/jobs/{job_id}/result").status_code == 200  # legacy endpoint still works
    assert client.get(f"/api/v1/jobs/{job_id}/segments/seg_00001/dub-audio").status_code == 200

    # promote Rahul's cloned voice into the library, reuse it for episode 2
    r = client.post(f"/api/v1/projects/{pid}/speakers/promote", json={"job_id": job_id, "job_speaker_id": "SPEAKER_00"})
    assert r.status_code == 201, r.text
    rahul = r.json()
    assert rahul["display_name"] == "Rahul" and rahul["source"] == "cloned" and rahul["reference_available"]
    calls_before = tp.FakeIndicF5.calls
    ep2 = video.with_name("episode 02.mp4")
    shutil.copyfile(video, ep2)
    job2 = _upload(client, pid, ep2)["job_id"]
    assert job2 != job_id
    _wait(client, job2, {"awaiting_review"})
    assert fake_models == asr_calls, "same media content: cached analysis is reused across videos"
    d = client.get(f"/api/v1/jobs/{job2}/draft").json()
    assert d["segments"][2]["translated_text"] == tp.HINDI["seg_00003"], "episode 2 has its own independent draft"
    r = client.patch(f"/api/v1/jobs/{job2}/draft", json={"base_version": 1, "speakers": [
        {"speaker_id": "SPEAKER_00", "display_name": "Rahul", "voice_source": "library",
         "library_speaker_id": rahul["speaker_id"]},
        {"speaker_id": "SPEAKER_01", "voice_source": "clone", "engine": "indicf5"}]})
    assert r.status_code == 200, r.text
    # the library speaker is in use by an open job -> cannot be deleted
    assert client.delete(f"/api/v1/projects/{pid}/speakers/{rahul['speaker_id']}").status_code == 409
    assert client.post(f"/api/v1/jobs/{job2}/confirm", json={"base_version": 2}).status_code == 202
    j2 = _wait(client, job2, {"succeeded", "failed"})
    assert j2["state"] == "succeeded", j2
    res2 = client.get(f"/api/v1/jobs/{job2}/voice-resolution").json()
    assert res2["SPEAKER_00"]["resolved_source"] == "library"
    assert res2["SPEAKER_00"]["library_speaker_id"] == rahul["speaker_id"]
    assert tp.FakeIndicF5.calls > calls_before, "a changed voice reference must re-synthesize"

    detail = client.get(f"/api/v1/projects/{pid}").json()
    assert detail["video_count"] == 2 and detail["speaker_count"] == 1 and detail["jobs"]["succeeded"] == 2
    assert client.get(f"/api/v1/jobs/{job_id}/logs", params={"tail": 5}).json()["lines"]


def test_cancel_and_retry(client, video, fake_models):
    pid = client.post("/api/v1/projects", json={"name": "C"}).json()["project_id"]
    job_id = _upload(client, pid, video)["job_id"]
    _wait(client, job_id, {"awaiting_review"})
    r = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert r.status_code == 200 and r.json()["state"] == "cancelled"
    assert client.post(f"/api/v1/jobs/{job_id}/cancel").status_code == 409  # terminal
    assert client.post(f"/api/v1/jobs/{job_id}/retry").status_code == 200
    job = _wait(client, job_id, {"awaiting_review", "failed"})
    assert job["state"] == "awaiting_review"
    assert client.post(f"/api/v1/jobs/{job_id}/retry").status_code == 409  # not failed


def test_edited_translation_only_resynthesizes_that_line(settings, video, fake_models, tmp_path):
    """Per-line TTS cache: changing one translation re-runs TTS for that line only."""
    pipe = pl.DubbingPipeline(settings)
    analysis = pipe.analyze(video, pl.JobOptions(mix_mode=pl.MixMode.DUB_ONLY), pl._Progress(lambda *a: None))
    requests = {
        speaker_id: VoiceRequest(speaker_id, speaker_id, VoiceSource.CAST,
                                 voice_prompt="A natural Hindi speaker with clear diction.")
        for speaker_id in {block["speaker"] for block in analysis.blocks}
    }
    plan = pl.GenerationPlan(blocks=analysis.blocks, translations=dict(analysis.translations),
                             source_language="en", voice_requests=requests)
    opts = pl.JobOptions(mix_mode=pl.MixMode.DUB_ONLY)
    pipe.generate_reviewed(video, analysis.turns, plan, opts, tmp_path / "a", lambda *a: None)
    first = tp.FakeParler.calls
    plan.translations["seg_00001"] = "नमस्ते जी"
    plan.locked = {"seg_00001"}
    pipe.generate_reviewed(video, analysis.turns, plan, opts, tmp_path / "b", lambda *a: None)
    assert tp.FakeParler.calls == first + 1


def test_job_state_transitions_and_old_records(settings):
    from backend.app.jobs import InvalidTransition, JobManager, JobRecord, JobState

    JobManager._check_transition(JobState.RUNNING, JobState.AWAITING_REVIEW)
    JobManager._check_transition(JobState.AWAITING_REVIEW, JobState.GENERATING)
    for old, new in [(JobState.SUCCEEDED, JobState.QUEUED), (JobState.QUEUED, JobState.GENERATING),
                     (JobState.AWAITING_REVIEW, JobState.SUCCEEDED), (JobState.FAILED, JobState.SUCCEEDED)]:
        with pytest.raises(InvalidTransition):
            JobManager._check_transition(old, new)
    # a record written by the previous version (no project/review fields) still loads
    old = {"job_id": "a" * 32, "state": "succeeded", "stage": "done", "progress": 1.0, "message": "",
           "input_filename": "x.mp4", "input_path": "x.mp4", "options": {}, "created_at": "2026-10-01T00:00:00Z"}
    rec = JobRecord.model_validate(old)
    assert rec.review_required is False and rec.phase == "analysis" and rec.project_id is None
