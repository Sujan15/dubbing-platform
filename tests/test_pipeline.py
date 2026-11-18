"""Pipeline tests that need only FFmpeg: model stages are replaced by deterministic fakes.

    .venv\\Scripts\\python -m pytest -q

They exercise everything the models do not: probing, extraction, stage caching, speaker
alignment, the timing policy (fit / atempo / LLM refine / capped overflow), loudness matching,
all three mix modes, muxing, output validation, the translation client and the HTTP API.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from backend.app.config import MixMode, Settings, TTSEngine, TranslationEngine
from backend.services import pipeline as pl
from backend.services import tts
from backend.services import translation as tr
from backend.services.media import FFmpeg
from backend.services.tts.base import BaseTTSEngine, EngineMetadata

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")

# Source dialogue: (start, end, speaker, text). Durations are chosen so the fake TTS lands in each timing branch.
DIALOGUE = [
    (0.50, 2.80, "SPEAKER_00", "Hello there, how are you?"),
    (3.00, 4.00, "SPEAKER_01", "I am fine."),
    (4.50, 5.50, "SPEAKER_00", "Run!"),
]
# Fake Hindi; fake TTS speaks 0.1 s per character.
HINDI = {"seg_00001": "नमस्ते, आप कैसे हैं?",   # 20 chars -> 2.0 s in 2.3 s  -> fits
         "seg_00002": "मैं ठीक हूँ जी",          # 14 chars -> 1.4 s in 1.0 s -> refine
         "seg_00003": "भागो"}                    # 4 chars  -> 0.4 s in 1.0 s -> fits


@pytest.fixture()
def video(tmp_path: Path) -> Path:
    out = tmp_path / "clip with spaces.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=7",
                    "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=7",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
    return out


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(_env_file=None, data_dir=tmp_path / "data", separation_enabled=False, device="cpu",
                 openrouter_api_key="test-key", hf_token="hf_test", tts_engines_enabled="parler,indicf5",
                 timing_max_refine_attempts=2)
    s.ensure_dirs()
    return s


class FakeTranslator:
    shorten_calls: list = []

    def __init__(self, settings, engine):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def close(self):
        pass

    def translate(self, lines, source_language, progress=None):
        if progress:
            progress(1.0)
        return {l.id: HINDI[l.id] for l in lines}

    def shorten(self, source_text, hindi, target_chars, duration, source_language):
        FakeTranslator.shorten_calls.append((hindi, target_chars))
        return hindi[:target_chars]


class _FakeTTSEngine(BaseTTSEngine):
    sample_rate = 24000
    fingerprint = {"fake": 1}
    calls = 0

    def __init__(self, settings, device):
        super().__init__(settings, device)

    @classmethod
    def check_model_availability(cls, settings):
        return {"available": True, "reason": "", "model_id": f"fake/{cls.engine.value}", "revision": "r1"}

    def get_model_metadata(self):
        return EngineMetadata(self.engine, f"fake/{self.engine.value}", "r1", self.sample_rate, self.device)

    def synthesize(self, text, spec, seed, speed=1.0):
        type(self).calls += 1
        n = int(0.1 * len(text) * self.sample_rate)
        t = np.arange(n) / self.sample_rate
        pad = np.zeros(int(0.3 * self.sample_rate), dtype=np.float32)  # trailing silence must be trimmed
        return np.concatenate([0.3 * np.sin(2 * np.pi * 300 * t).astype(np.float32), pad])


class FakeParler(_FakeTTSEngine):
    engine = TTSEngine.PARLER
    capabilities = tts.engine_class(TTSEngine.PARLER).capabilities


class FakeIndicF5(_FakeTTSEngine):
    engine = TTSEngine.INDICF5
    capabilities = tts.engine_class(TTSEngine.INDICF5).capabilities


@pytest.fixture()
def fake_models(monkeypatch):
    def transcribe(self, audio, out, language, duration, p):
        segs = [{"start": s, "end": e, "text": t, "avg_logprob": -0.1, "no_speech_prob": 0.01,
                 "words": [{"start": s, "end": e, "word": " " + t, "probability": 0.9}]} for s, e, _, t in DIALOGUE]
        pl._write_json(out / "transcript.json", {"language": "en", "language_probability": 0.99, "segments": segs})

    def diarize(self, audio, out, options, p):
        pl._write_json(out / "diarization.json", [{"start": s, "end": e, "speaker": spk} for s, e, spk, _ in DIALOGUE])

    monkeypatch.setattr(pl.DubbingPipeline, "_transcribe", transcribe)
    monkeypatch.setattr(pl.DubbingPipeline, "_diarize", diarize)
    monkeypatch.setattr(pl, "Translator", FakeTranslator)
    monkeypatch.setitem(tts._REGISTRY, TTSEngine.PARLER, FakeParler)
    monkeypatch.setitem(tts._REGISTRY, TTSEngine.INDICF5, FakeIndicF5)
    FakeTranslator.shorten_calls = []
    FakeParler.calls = 0
    FakeIndicF5.calls = 0


@pytest.mark.parametrize("mode", list(MixMode))
def test_end_to_end(video, settings, fake_models, tmp_path, mode):
    pipe = pl.DubbingPipeline(settings)
    events = []
    result = pipe.run(video, pl.JobOptions(mix_mode=mode), tmp_path / "job", lambda *a: events.append(a))

    ff = FFmpeg("ffmpeg", "ffprobe", 600)
    src, out = ff.probe(video), ff.probe(result.output_video)
    assert abs(out.duration - src.duration) < 0.5 and out.audio_streams == 1 and out.video_codec == "h264"
    assert [e[1] for e in events] == sorted(e[1] for e in events) and events[-1][1] == pytest.approx(1.0)

    manifest = [pl.DialogueSegment(**m) for m in json.loads(result.manifest.read_text(encoding="utf-8"))]
    assert [m.speaker_id for m in manifest] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    for m in manifest:
        assert Path(m.generated_audio_path).is_file()
        assert m.generated_duration <= m.original_duration * (1 + settings.timing_atempo_threshold) + 0.05

    report = {r["segment_id"]: r for r in json.loads(result.timing_report.read_text(encoding="utf-8"))["segments"]}
    assert report["seg_00001"]["status"] == "fits"
    assert report["seg_00002"]["refine_attempts"] >= 1 and FakeTranslator.shorten_calls
    assert manifest[1].translated_text != HINDI["seg_00002"]  # refined text is what was spoken
    if mode == MixMode.PRESERVE_BACKGROUND:  # no stems without Demucs -> documented fallback
        assert any("original_plus_dub" in w for w in result.warnings)
    qa = json.loads(result.qa_report.read_text(encoding="utf-8"))
    assert qa["clipping"]["clipped_samples"] == 0
    assert qa["clipping"]["peak"] < 0.999


def test_stages_are_cached(video, settings, fake_models, tmp_path):
    pipe = pl.DubbingPipeline(settings)
    pipe.run(video, pl.JobOptions(), tmp_path / "a", lambda *a: None)
    calls_first = FakeParler.calls
    t0 = time.time()
    pipe.run(video, pl.JobOptions(), tmp_path / "b", lambda *a: None)
    assert FakeParler.calls == calls_first, "second run must not re-synthesize"
    assert time.time() - t0 < 30
    ws = next(settings.work_dir.iterdir())
    assert not list(ws.rglob(".tmp-*")), "no temporary stage directories may survive"


def test_timing_overflow_is_capped(video, settings, fake_models, tmp_path, monkeypatch):
    monkeypatch.setattr(FakeTranslator, "shorten", lambda self, s, h, t, d, l: h)  # LLM cannot shorten
    result = pl.DubbingPipeline(settings).run(video, pl.JobOptions(mix_mode=MixMode.DUB_ONLY), tmp_path / "j",
                                              lambda *a: None)
    seg = json.loads(result.timing_report.read_text(encoding="utf-8"))["segments"][1]
    assert seg["status"] == "atempo_capped_overflow" and seg["refine_attempts"] == 2
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest[1]["speed_ratio"] == pytest.approx(settings.timing_max_atempo)
    assert result.qa_status == "FAILED"


def test_mix_peak_protection_handles_overlapping_lines(settings, tmp_path):
    sr = settings.mix_sample_rate
    master = tmp_path / "master.wav"
    sf.write(master, np.zeros(sr * 2, dtype=np.float32), sr, subtype="FLOAT")
    t = np.arange(sr, dtype=np.float32) / sr
    clip = tmp_path / "line.wav"
    sf.write(clip, 0.95 * np.sin(2 * np.pi * 440 * t), sr, subtype="FLOAT")
    segments = [pl.DialogueSegment(
        segment_id=f"line_{i}", start_time=0.0, end_time=1.0, speaker_id=f"S{i}",
        original_text="line", translated_text="line", generated_audio_path=str(clip), original_duration=1.0,
    ) for i in range(3)]
    out = tmp_path / "mix"
    out.mkdir()
    pl.DubbingPipeline(settings)._mix(segments, MixMode.DUB_ONLY, master, None, out, lambda *args: None)
    mixed = sf.read(out / "mix.wav", dtype="float32")[0]
    peak = float(np.max(np.abs(mixed)))
    report = json.loads((out / "mix_report.json").read_text(encoding="utf-8"))
    assert report["peak_protection"]["input_peak"] > 1.0
    assert peak <= 10 ** ((settings.true_peak_ceiling_dbtp - 0.5) / 20) + 1e-5


def test_build_dialogue_blocks_splits_speakers_inside_one_segment():
    seg = {"start": 0.0, "end": 4.0, "text": "a b c d", "words": [
        {"start": 0.0, "end": 0.9, "word": " Where"}, {"start": 1.0, "end": 1.9, "word": " are"},
        {"start": 2.2, "end": 3.0, "word": " Home"}, {"start": 3.1, "end": 3.9, "word": " now"}]}
    turns = [{"start": 0.0, "end": 2.0, "speaker": "A"}, {"start": 2.1, "end": 4.0, "speaker": "B"}]
    blocks = pl.build_dialogue_blocks([seg], turns)
    assert [(b["speaker"], b["text"]) for b in blocks] == [("A", "Where are"), ("B", "Home now")]
    # a 0.1 s single-word blip of another speaker is absorbed
    turns2 = [{"start": 0.0, "end": 2.15, "speaker": "A"}, {"start": 2.15, "end": 2.3, "speaker": "B"},
              {"start": 2.3, "end": 4.0, "speaker": "A"}]
    assert len(pl.build_dialogue_blocks([seg], turns2)) == 1


def test_ducking_envelope():
    env = pl.ducking_envelope(0, 1000, [(400, 600, "x")], gain=0.1, ramp=100)
    assert env[0] == 1.0 and env[500] == pytest.approx(0.1) and 0.1 < env[350] < 1.0 and env[999] == 1.0


def test_select_reference_spans_prefers_solo_speech():
    blocks = [{"start": 0, "end": 6, "speaker": "A"}, {"start": 10, "end": 18, "speaker": "A"},
              {"start": 20, "end": 21, "speaker": "B"}]
    turns = [{"start": 0, "end": 6, "speaker": "A"}, {"start": 3, "end": 5, "speaker": "B"},
             {"start": 10, "end": 18, "speaker": "A"}]
    assert pl.select_reference_spans(blocks, turns, "A", 10.0) == [[10, 18]]


def test_openrouter_translator_retries_and_validates(settings, monkeypatch):
    monkeypatch.setattr(tr.time, "sleep", lambda s: None)
    responses = iter([
        httpx.Response(429, text="slow down"),
        httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
            {"translations": [{"id": "s1", "text": "hello"}]})}}]}),  # not Devanagari -> split & retry
        httpx.Response(200, json={"choices": [{"message": {"content": "```json\n" + json.dumps(
            {"translations": [{"id": "s1", "text": "नमस्ते"}]}, ensure_ascii=False) + "\n```"}}]}),
    ])
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return next(responses)

    t = tr.Translator(settings, TranslationEngine.OPENROUTER)
    t.clients[0]._http = httpx.Client(base_url="https://x", transport=httpx.MockTransport(handler))
    with pytest.raises(tr.TranslationError):  # single-line batch with bad text cannot be split further
        t.translate([tr.SourceLine("s1", "A", "Hello", 1.0)], "English")
    assert t.translate([tr.SourceLine("s1", "A", "Hello", 1.0)], "English") == {"s1": "नमस्ते"}
    assert "natural, idiomatic Hindi suitable for film dubbing" in seen[0]["messages"][0]["content"]


def test_api_rejects_bad_upload(settings, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app import config, main

    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    with TestClient(main.app) as client:
        assert client.get("/api/v1/health").json()["status"] == "ok"
        r = client.post("/api/v1/jobs", files={"file": ("notes.txt", b"hi", "text/plain")})
        assert r.status_code == 415
        r = client.post("/api/v1/jobs", files={"file": ("fake.mp4", b"not a video", "video/mp4")})
        assert r.status_code == 422
        assert client.get("/api/v1/jobs/nope").status_code == 404


def test_api_job_lifecycle(video, settings, fake_models, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app import config, main

    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    with TestClient(main.app) as client:
        with open(video, "rb") as f:
            r = client.post("/api/v1/jobs", files={"file": (video.name, f, "video/mp4")},
                            data={"source_language": "en", "mix_mode": "dub_only"})
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]
        assert client.get(f"/api/v1/jobs/{job_id}/result").status_code in (409, 200)
        for _ in range(240):
            job = client.get(f"/api/v1/jobs/{job_id}").json()
            if job["state"] in ("succeeded", "failed"):
                break
            time.sleep(0.25)
        assert job["state"] == "succeeded", job
        r = client.get(f"/api/v1/jobs/{job_id}/result")
        assert r.status_code == 200 and len(r.content) > 1000
        assert (settings.outputs_dir / job_id / "job.log").stat().st_size > 0
