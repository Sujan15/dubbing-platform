"""v3 tests: engine registry, hybrid Voice Library (audition -> approve gate), series characters with
permanent voice profiles reused across episodes, speaker identity suggestions and the short-dialogue
policy, separation fallback policies, dynamic ducking, the QA gate and backward compatibility of v2
records. Models are deterministic fakes; only FFmpeg is real.

    .venv\\Scripts\\python -m pytest -q
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import test_pipeline as tp
import test_projects as tpr
from fastapi.testclient import TestClient

from backend.app import config, main
from backend.app.config import MixMode, SeparationFallback, Settings, TTSEngine
from backend.services import drafts as dr
from backend.services import pipeline as pl
from backend.services import speaker_identity as si
from backend.services import tts
from backend.services.mixing import DuckingFollower, plan_gains
from backend.services.tts.base import BaseTTSEngine, EngineMetadata, VoiceSpec
from backend.services.voices import VoiceRequest, VoiceResolutionError, VoiceSource, resolve_voice

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
video = tpr.video
fake_models = tpr.fake_models
_wait, _upload = tpr._wait, tpr._upload


# ------------------------------------------------------------------------------- fakes
class _FakeEngine(BaseTTSEngine):
    """Deterministic engine: a tone whose frequency depends on the voice identity (prompt or reference)."""
    sample_rate = 24000
    fingerprint = {"fake": 1}
    calls: list = []

    @classmethod
    def check_model_availability(cls, settings):
        return {"available": True, "reason": "", "model_id": f"fake/{cls.engine.value}", "revision": "r1"}

    def __init__(self, settings, device):
        super().__init__(settings, device)

    def get_model_metadata(self):
        return EngineMetadata(self.engine, f"fake/{self.engine.value}", "r1", self.sample_rate, self.device)

    def synthesize(self, text, spec, seed, speed=1.0):
        _FakeEngine.calls.append((self.engine.value, text, spec.voice_prompt, str(spec.reference_path)))
        ident = spec.voice_prompt or str(spec.reference_path)
        freq = 200 + int(hashlib.sha256(ident.encode()).hexdigest()[:2], 16)
        n = int(0.1 * len(text) * self.sample_rate)
        t = np.arange(n) / self.sample_rate
        return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class FakeParler(_FakeEngine):
    engine = TTSEngine.PARLER
    capabilities = tts.engine_class(TTSEngine.PARLER).capabilities


class FakeIndicF5(_FakeEngine):
    engine = TTSEngine.INDICF5
    capabilities = tts.engine_class(TTSEngine.INDICF5).capabilities


class FakeEmbedder:
    """Embedding = hash of the reference file content: identical speech -> identical identity."""
    def __init__(self, settings, device):
        pass

    @staticmethod
    def available():
        return True

    def embed_file(self, path, sample_rate=16000):
        h = hashlib.sha256(Path(path).read_bytes()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        return si.unit(rng.standard_normal(32).astype(np.float32))

    def unload(self):
        pass


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = Settings(_env_file=None, data_dir=tmp_path / "data", separation_enabled=False, device="cpu",
                 openrouter_api_key="test-key", hf_token="hf_test",
                 timing_max_refine_attempts=2, min_speaker_reference_seconds=2.0,
                 tts_engines_enabled="parler,indicf5", min_segment_identity_duration=1.5,
                 speaker_match_high_threshold=0.8, speaker_match_review_threshold=0.5)
    s.ensure_dirs()
    tpr._tone(s.default_voices_dir / "narrator.wav", 6.0)
    (s.default_voices_dir / "narrator.json").write_text(json.dumps({"label": "Narrator", "gender": "male"}))
    return s


@pytest.fixture()
def fake_engines(monkeypatch, fake_models):
    monkeypatch.setitem(tts._REGISTRY, TTSEngine.PARLER, FakeParler)
    monkeypatch.setitem(tts._REGISTRY, TTSEngine.INDICF5, FakeIndicF5)
    monkeypatch.setattr(pl, "SpeakerEmbedder", FakeEmbedder)
    _FakeEngine.calls = []
    return fake_models


@pytest.fixture()
def client(settings, monkeypatch):
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    with TestClient(main.app) as c:
        yield c


def _audition(client, voice_id: str, text: str = "नमस्ते दुनिया") -> dict:
    r = client.post(f"/api/v1/voice-library/{voice_id}/audition", json={"text": text})
    assert r.status_code == 202, r.text
    aid = r.json()["audition_id"]
    for _ in range(200):
        e = client.get(f"/api/v1/voice-library/auditions/{aid}").json()
        if e["status"] in ("done", "failed"):
            return e
        time.sleep(0.05)
    raise AssertionError("audition stuck")


# ------------------------------------------------------------------------------- engines
def test_default_tts_engines_are_parler_primary_and_indicf5_secondary(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    assert settings.enabled_tts_engines == {TTSEngine.PARLER, TTSEngine.INDICF5}


def test_engine_registry_status_and_gating(settings, fake_engines):
    status = {e["engine"]: e for e in tts.engine_status(settings)}
    assert set(status) == {"parler", "indicf5"}
    assert status["parler"]["capabilities"]["prompt_conditioned"] and not status["parler"]["capabilities"]["reference_conditioned"]
    assert status["indicf5"]["capabilities"]["needs_reference_transcript"]
    off = Settings(_env_file=None, data_dir=settings.data_dir, tts_engines_enabled="indicf5")
    with pytest.raises(tts.EngineUnavailable):
        tts.load_engine(off, "cpu", TTSEngine.PARLER)
    with pytest.raises(ValueError):
        Settings(_env_file=None, data_dir=settings.data_dir, tts_engines_enabled="xtts")
    with pytest.raises(ValueError):
        Settings(_env_file=None, data_dir=settings.data_dir, tts_engines_enabled="xtts,bogus")


def test_voice_spec_cache_fingerprint_covers_prompt_model_and_seed():
    a = VoiceSpec(engine=TTSEngine.PARLER, voice_prompt="warm male", seed=1).cache_fingerprint(None)
    b = VoiceSpec(engine=TTSEngine.PARLER, voice_prompt="warm male", seed=2).cache_fingerprint(None)
    c = VoiceSpec(engine=TTSEngine.PARLER, voice_prompt="deep male", seed=1).cache_fingerprint(None)
    assert a != b != c and a != c


def test_indicf5_clone_requires_transcript(settings):
    bank = pl.DefaultVoiceBank(settings)
    ref = tpr._tone(settings.data_dir / "ref.wav", 8.0)
    req = VoiceRequest("S", "Hero", VoiceSource.CLONE, engine=TTSEngine.INDICF5)
    with pytest.raises(VoiceResolutionError):
        resolve_voice(req, ref, 8.0, None, bank, settings)
    rv = resolve_voice(req, ref, 8.0, None, bank, settings, clone_transcript="Hello there")
    assert rv.engine == TTSEngine.INDICF5 and rv.spec.reference_transcript == "Hello there"
    with pytest.raises(VoiceResolutionError, match="explicitly select IndicF5"):
        resolve_voice(VoiceRequest("S", "Hero"), ref, 8.0, None, bank, settings,
                      clone_transcript="Hello there")


# ------------------------------------------------------------------------------- voice library
def test_voice_library_cast_preset_audition_and_approval_gate(client, fake_engines):
    engine_defaults = client.get("/api/v1/engines").json()
    assert engine_defaults["default_engine"] == "parler"
    assert engine_defaults["default_clone_engine"] == "indicf5"

    r = client.post("/api/v1/voice-library/catalog/seed")
    assert r.status_code == 201 and len(r.json()["created"]) == 5
    slots = client.get("/api/v1/voice-library", params={"engine": "parler"}).json()["voices"]
    by_slot = {v["catalog_slot"]: v for v in slots}
    assert set(by_slot) == {
        "HIN_MALE_HERO_01",
        "HIN_FEMALE_HEROINE_01",
        "HIN_MALE_VILLAIN_01",
        "HIN_FEMALE_SUPPORT_01",
        "HIN_MALE_SUPPORT_01",
    }
    assert all(not v["approved"] and not v["active"] and not v["selectable"] for v in slots)
    expected_prompts = {
        "HIN_MALE_HERO_01": (
            "Rohit speaks with a balanced, confident male pitch and a natural, expressive tone at a moderate pace. "
            "The recording is of very high quality, sounding very clear and close up with almost no background noise."
        ),
        "HIN_FEMALE_HEROINE_01": (
            "Divya speaks with a warm, slightly high-pitched female voice and an expressive, engaging tone at a "
            "moderate pace. The recording is very clear audio, close-sounding, and captured in a studio environment "
            "with no background noise."
        ),
        "HIN_MALE_VILLAIN_01": (
            "Aman's deep, low-pitched male voice delivers speech with an intense, slightly slower, and authoritative "
            "tone. The recording is very clear audio, close-sounding, with no background noise."
        ),
        "HIN_FEMALE_SUPPORT_01": (
            "Rani speaks with a natural female pitch and a calm, conversational, slightly expressive tone at a normal "
            "pace. The recording is of very high quality, very clear audio, and close-sounding with no background noise."
        ),
        "HIN_MALE_SUPPORT_01": (
            "Rohit speaks at a moderate pace with a neutral, conversational male pitch and a slightly monotone, steady "
            "delivery. The recording is very clear audio, close-sounding, with no background noise."
        ),
    }
    assert {v["catalog_slot"]: v["voice_description_prompt"] for v in slots} == expected_prompts
    assert {v["catalog_slot"]: v["voice_name"] for v in slots} == {
        "HIN_MALE_HERO_01": "Hero — Rohit",
        "HIN_FEMALE_HEROINE_01": "Heroine — Divya",
        "HIN_MALE_VILLAIN_01": "Villain — Aman",
        "HIN_FEMALE_SUPPORT_01": "Supporting Female — Rani",
        "HIN_MALE_SUPPORT_01": "Supporting Male — Rohit",
    }
    assert len({v["voice_id"] for v in slots}) == 5
    assert all(v["prompt_version"] == 1 for v in slots)
    reseed = client.post("/api/v1/voice-library/catalog/seed").json()
    assert reseed["created"] == [] and reseed["updated"] == []
    reseeded_slots = client.get("/api/v1/voice-library", params={"engine": "parler"}).json()["voices"]
    assert {v["catalog_slot"]: v["voice_id"] for v in reseeded_slots} == {
        v["catalog_slot"]: v["voice_id"] for v in slots
    }

    for voice in slots:
        audition = _audition(client, voice["voice_id"])
        assert audition["status"] == "done", audition
        auditioned = client.get(f"/api/v1/voice-library/{voice['voice_id']}").json()
        assert auditioned["preview"]["artifact_name"]
        assert auditioned["preview"]["prompt_version"] == voice["prompt_version"]
        assert not auditioned["approved"] and not auditioned["selectable"]

    r = client.post("/api/v1/voice-library", json={"voice_name": "Hero Cast", "engine": "parler", "gender": "male",
                                                   "voice_description_prompt": "A confident warm male Hindi voice"})
    assert r.status_code == 201, r.text
    v = r.json()
    assert v["kind"] == "cast" and v["model_id"] == "ai4bharat/indic-parler-tts" and not v["selectable"]
    assert v["prompt_version"] == 1 and len(v["prompt_versions"]) == 1
    defaulted = client.post("/api/v1/voice-library", json={
        "voice_name": "Default Parler",
        "voice_description_prompt": "A clear Hindi voice",
    })
    assert defaulted.status_code == 201 and defaulted.json()["engine"] == "parler"
    # cannot approve without an audition of the current configuration
    assert client.post(f"/api/v1/voice-library/{v['voice_id']}/approval", json={"approved": True}).status_code == 409
    assert client.post("/api/v1/voice-library", json={"voice_name": "No prompt", "engine": "parler"}).status_code == 422

    e = _audition(client, v["voice_id"])
    assert e["status"] == "done", e
    assert _FakeEngine.calls and _FakeEngine.calls[-1][0] == "parler"
    assert client.get(f"/api/v1/voice-library/{v['voice_id']}/preview").status_code == 200
    assert client.get(f"/api/v1/voice-library/{v['voice_id']}/preview",
                      params={"run_id": e["audition_id"]}).status_code == 200
    v = client.get(f"/api/v1/voice-library/{v['voice_id']}").json()
    assert v["preview"]["model_id"] == "fake/parler" and v["preview"]["model_revision"] == "r1"
    assert v["preview"]["prompt_version"] == 1 and v["preview"]["artifact_name"]
    r = client.post(f"/api/v1/voice-library/{v['voice_id']}/approval", json={"approved": True, "note": "clear Hindi"})
    assert r.status_code == 200 and r.json()["selectable"] and r.json()["supported_language_status"] == "validated"

    # changing the prompt invalidates approval until a new audition
    r = client.patch(f"/api/v1/voice-library/{v['voice_id']}", json={"voice_description_prompt": "A deep slow villain"})
    assert r.status_code == 200 and not r.json()["approved"] and not r.json()["selectable"]
    assert r.json()["prompt_version"] == 2 and len(r.json()["prompt_versions"]) == 2
    assert client.post(f"/api/v1/voice-library/{v['voice_id']}/approval", json={"approved": True}).status_code == 409
    second_audition = _audition(client, v["voice_id"])
    assert client.post(f"/api/v1/voice-library/{v['voice_id']}/approval", json={"approved": True}).json()["selectable"]
    assert client.get(f"/api/v1/voice-library/{v['voice_id']}/preview",
                      params={"run_id": e["audition_id"]}).status_code == 200
    assert client.get(f"/api/v1/voice-library/{v['voice_id']}/preview",
                      params={"run_id": second_audition["audition_id"]}).status_code == 200
    runs = client.get(f"/api/v1/voice-library/{v['voice_id']}/runs").json()["runs"]
    assert len(runs) == 2 and runs[0]["kind"] == "audition" and runs[0]["model_id"] == "fake/parler"

    # filters + legacy voices in the unified listing
    lib = client.get("/api/v1/voice-library", params={"approved": True, "include_legacy": True}).json()["voices"]
    assert {x["voice_name"] for x in lib} >= {"Hero Cast", "Narrator"}
    assert any(x["legacy"] and x["engine"] == "xtts" for x in lib)
    assert client.get("/api/v1/voice-library", params={"gender": "female", "include_legacy": False}).json()["voices"]


def test_catalog_seed_refreshes_existing_slot_without_replacing_its_id(client):
    first = client.post("/api/v1/voice-library/catalog/seed").json()
    hero = next(v for v in first["created"] if v["catalog_slot"] == "HIN_MALE_HERO_01")
    response = client.patch(f"/api/v1/voice-library/{hero['voice_id']}",
                            json={"voice_description_prompt": "Outdated catalog prompt"})
    assert response.status_code == 200 and response.json()["prompt_version"] == 2

    refreshed = client.post("/api/v1/voice-library/catalog/seed").json()
    assert refreshed["created"] == []
    assert len(refreshed["updated"]) == 1
    updated = refreshed["updated"][0]
    assert updated["voice_id"] == hero["voice_id"]
    assert updated["prompt_version"] == 3
    assert updated["voice_description_prompt"] == (
        "Rohit speaks with a balanced, confident male pitch and a natural, expressive tone at a moderate pace. "
        "The recording is of very high quality, sounding very clear and close up with almost no background noise."
    )


def test_voice_library_indicf5_clone_reference_and_replace_rule(client, fake_engines, tmp_path):
    r = client.post("/api/v1/voice-library", json={"voice_name": "Meera Clone", "engine": "indicf5", "gender": "female"})
    assert r.status_code == 201
    vid = r.json()["voice_id"]
    wav = tpr._tone(tmp_path / "meera.wav", 12.0)
    with open(wav, "rb") as f:  # IndicF5 needs the transcript
        assert client.post(f"/api/v1/voice-library/{vid}/reference", files={"file": ("m.wav", f, "audio/wav")}).status_code == 422
    with open(wav, "rb") as f:
        r = client.post(f"/api/v1/voice-library/{vid}/reference", files={"file": ("m.wav", f, "audio/wav")},
                        data={"transcript": "मेरा नाम मीरा है"})
    assert r.status_code == 200, r.text
    assert r.json()["reference_available"] and r.json()["reference_sha256"]
    with open(wav, "rb") as f:  # an existing master reference is never replaced silently
        assert client.post(f"/api/v1/voice-library/{vid}/reference", files={"file": ("m.wav", f, "audio/wav")},
                           data={"transcript": "x"}).status_code == 409
    assert _audition(client, vid)["status"] == "done"
    assert _FakeEngine.calls[-1][0] == "indicf5"
    assert client.post(f"/api/v1/voice-library/{vid}/approval", json={"approved": True}).json()["selectable"]


# ------------------------------------------------------------------------------- series + identity
def test_series_characters_permanent_voices_and_cross_episode_reuse(client, settings, video, fake_engines, tmp_path):
    pid = client.post("/api/v1/projects", json={"name": "Shadow Kingdom", "kind": "series"}).json()["project_id"]
    arjun = client.post(f"/api/v1/projects/{pid}/characters", json={"canonical_name": "Arjun", "role": "hero",
                                                                    "aliases": ["Arj"]}).json()
    meera = client.post(f"/api/v1/projects/{pid}/characters", json={"canonical_name": "Meera", "role": "heroine"}).json()
    assert client.post(f"/api/v1/projects/{pid}/characters", json={"canonical_name": "arj"}).status_code == 409
    assert not arjun["has_approved_voice"]

    # permanent voices: Arjun -> approved Parler cast voice, Meera -> transcript-bearing IndicF5 voice
    cast = client.post("/api/v1/voice-library", json={"voice_name": "HERO", "engine": "parler",
                                                      "voice_description_prompt": "confident male"}).json()
    assert client.post(f"/api/v1/projects/{pid}/characters/{arjun['character_id']}/voice-profiles",
                       json={"voice_source": "cast", "voice_asset_id": cast["voice_id"]}).status_code == 422  # unapproved
    _audition(client, cast["voice_id"])
    client.post(f"/api/v1/voice-library/{cast['voice_id']}/approval", json={"approved": True})
    r = client.post(f"/api/v1/projects/{pid}/characters/{arjun['character_id']}/voice-profiles",
                    json={"voice_source": "cast", "voice_asset_id": cast["voice_id"], "label": "HERO_001"})
    assert r.status_code == 201 and r.json()["has_approved_voice"]
    profile_id = r.json()["active_voice_profile_id"]
    client.patch(f"/api/v1/projects/{pid}/characters/{arjun['character_id']}/voice-profiles/{profile_id}",
                 json={"identity_locked": True})
    assert client.post(f"/api/v1/projects/{pid}/characters/{arjun['character_id']}/voice-profiles",
                       json={"voice_source": "cast", "voice_asset_id": cast["voice_id"]}).status_code == 409  # locked

    meera_voice = client.post("/api/v1/voice-library", json={
        "voice_name": "Meera Clone", "engine": "indicf5", "gender": "female"}).json()
    meera_ref = tpr._tone(settings.data_dir / "meera-reference.wav", 8.0)
    with open(meera_ref, "rb") as f:
        uploaded = client.post(f"/api/v1/voice-library/{meera_voice['voice_id']}/reference",
                               files={"file": ("meera.wav", f, "audio/wav")},
                               data={"transcript": "मेरा नाम मीरा है"})
    assert uploaded.status_code == 200, uploaded.text
    assert _audition(client, meera_voice["voice_id"])["status"] == "done"
    assert client.post(f"/api/v1/voice-library/{meera_voice['voice_id']}/approval",
                       json={"approved": True}).json()["selectable"]
    profile = client.post(f"/api/v1/projects/{pid}/characters/{meera['character_id']}/voice-profiles",
                          json={"voice_source": "cast", "voice_asset_id": meera_voice["voice_id"]})
    assert profile.status_code == 201 and profile.json()["has_approved_voice"]
    assert client.delete(f"/api/v1/voice-library/{cast['voice_id']}").status_code == 409  # in use by a profile

    # episode 1: nothing to suggest yet -> every speaker unassigned, user maps them
    up1 = _upload(client, pid, video, title="E01", episode_number="1")
    j1 = up1["job_id"]
    assert _wait(client, j1, {"awaiting_review", "failed"})["state"] == "awaiting_review"
    d = client.get(f"/api/v1/jobs/{j1}/draft").json()
    by = {s["speaker_id"]: s for s in d["speakers"]}
    assert by["SPEAKER_00"]["mapping_status"] == "unassigned" and "character" in by["SPEAKER_00"]["available_voice_options"]
    assert by["SPEAKER_01"]["mapping_status"] == "unassigned" and "clean speech" in by["SPEAKER_01"]["identity_reason"]
    assert (settings.outputs_dir / j1 / "identity_suggestions.json").is_file()
    assert any("Short line" in w for w in next(s for s in d["segments"] if s["segment_id"] == "seg_00003")["warnings"])
    assert d["segments"][0]["speaker_confidence"] == 1.0

    assert client.patch(f"/api/v1/jobs/{j1}/draft", json={"base_version": 1, "speakers": [
        {"speaker_id": "SPEAKER_00", "character_id": "f" * 32}]}).status_code == 422  # unknown character
    r = client.patch(f"/api/v1/jobs/{j1}/draft", json={"base_version": 1, "speakers": [
        {"speaker_id": "SPEAKER_00", "display_name": "Arjun", "voice_source": "character", "character_id": arjun["character_id"]},
        {"speaker_id": "SPEAKER_01", "display_name": "Meera", "voice_source": "character", "character_id": meera["character_id"]}]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["speakers"][0]["mapping_status"] == "confirmed" and d["speakers"][0]["character_name"] == "Arjun"
    r = client.post(f"/api/v1/jobs/{j1}/confirm", json={"base_version": 2})
    assert r.status_code == 202, r.text
    job = _wait(client, j1, {"succeeded", "failed"})
    assert job["state"] == "succeeded", job
    assert job["qa_status"] in ("PASS", "WARNING", "REVIEW_REQUIRED") and "parler" in job["engines"]
    res = client.get(f"/api/v1/jobs/{j1}/voice-resolution").json()
    assert res["SPEAKER_00"]["engine"] == "parler" and res["SPEAKER_00"]["character_id"] == arjun["character_id"]
    assert res["SPEAKER_00"]["voice_profile_id"] == profile_id and res["SPEAKER_00"]["requested_source"] == "character"
    assert res["SPEAKER_01"]["engine"] == "indicf5" and res["SPEAKER_01"]["resolved_source"] == "cast"
    runs = client.get(f"/api/v1/jobs/{j1}/generation-runs").json()["runs"]
    assert {r_["engine"] for r_ in runs} == {"parler", "indicf5"}
    assert all(r_["model_id"] and r_["text_hash"] for r_ in runs)
    assert next(r_ for r_ in runs if r_["speaker_id"] == "SPEAKER_00")["voice_profile_id"] == profile_id
    mapping = client.get(f"/api/v1/projects/{pid}/videos/{up1['video_id']}/speaker-mapping").json()
    assert {m["local_speaker_id"]: m["character_id"] for m in mapping["mappings"]} == {
        "SPEAKER_00": arjun["character_id"], "SPEAKER_01": meera["character_id"]}
    assert all(m["mapping_status"] == "confirmed" and m["reviewed_by_user"] for m in mapping["mappings"])
    arjun_now = client.get(f"/api/v1/projects/{pid}/characters/{arjun['character_id']}").json()
    assert arjun_now["embedding_count"] == 1 and arjun_now["episodes"][0]["episode_number"] == 1

    # episode 2: same voices -> Arjun suggested for his local speaker, Meera unassigned (too little speech)
    ep2 = video.with_name("episode 02.mp4")
    shutil.copyfile(video, ep2)
    up2 = _upload(client, pid, ep2, episode_number="2")
    j2 = up2["job_id"]
    _wait(client, j2, {"awaiting_review"})
    d = client.get(f"/api/v1/jobs/{j2}/draft").json()
    by = {s["speaker_id"]: s for s in d["speakers"]}
    assert by["SPEAKER_00"]["mapping_status"] == "suggested"
    assert by["SPEAKER_00"]["suggested_character_id"] == arjun["character_id"]
    assert by["SPEAKER_00"]["character_id"] is None, "a suggestion is never an assignment"
    assert by["SPEAKER_00"]["identity_confidence"] > 0.99 and by["SPEAKER_00"]["previous_episode_matches"]
    assert by["SPEAKER_01"]["mapping_status"] == "unassigned"
    preview = client.get(f"/api/v1/jobs/{j2}/confirm-preview").json()
    assert any("suggestion not confirmed" in w for w in preview["warnings"])
    calls_before = len(_FakeEngine.calls)
    r = client.patch(f"/api/v1/jobs/{j2}/draft", json={"base_version": 1, "speakers": [
        {"speaker_id": "SPEAKER_00", "voice_source": "character", "character_id": arjun["character_id"]},
        {"speaker_id": "SPEAKER_01", "voice_source": "character", "character_id": meera["character_id"]}]})
    assert r.status_code == 200, r.text
    assert client.post(f"/api/v1/jobs/{j2}/confirm", json={"base_version": 2}).status_code == 202
    assert _wait(client, j2, {"succeeded", "failed"})["state"] == "succeeded"
    res2 = client.get(f"/api/v1/jobs/{j2}/voice-resolution").json()
    assert res2["SPEAKER_00"]["voice_profile_id"] == profile_id and res2["SPEAKER_00"]["voice_id"] == cast["voice_id"]
    assert len(_FakeEngine.calls) == calls_before, "identical lines + identical voice profile -> fully cached"
    assert client.get(f"/api/v1/projects/{pid}/characters/{arjun['character_id']}").json()["embedding_count"] == 2
    log = (settings.projects_dir / pid / "characters" / arjun["character_id"] / "generation_log.jsonl").read_text()
    assert log.count("\n") == 2
    assert client.delete(f"/api/v1/projects/{pid}/characters/{arjun['character_id']}").status_code == 204
    assert client.get(f"/api/v1/projects/{pid}").json()["kind"] == "series"


def test_match_speakers_thresholds_and_short_dialogue_policy(settings):
    rng = np.random.default_rng(0)
    a, b = si.unit(rng.standard_normal(16)), si.unit(rng.standard_normal(16))
    chars = {"ARJUN": a[None], "MEERA": b[None]}
    near_a = si.unit(a + 0.15 * rng.standard_normal(16))
    out = {s.local_speaker_id: s for s in si.match_speakers(
        {"S0": a, "S1": near_a, "S2": si.unit(rng.standard_normal(16)), "S3": a},
        {"S0": 10.0, "S1": 10.0, "S2": 10.0, "S3": 0.8, "S4": 5.0}, chars, settings)}
    assert out["S0"].status == "suggested" and out["S0"].character_id == "ARJUN"
    assert out["S1"].status in ("suggested", "review") and out["S1"].character_id == "ARJUN"
    assert out["S2"].status == "unassigned" and out["S2"].character_id is None
    assert out["S3"].status == "unassigned" and "too little evidence" in out["S3"].reason  # identical voice, 0.8 s
    assert out["S4"].status == "unassigned" and "no embedding" in out["S4"].reason
    assert si.calibrate([(0.9, True), (0.4, False), (0.7, True), (0.6, False)])["table"]
    conf, flags = si.attribution_confidence({"start": 0, "end": 2, "speaker": "A"},
                                            [{"start": 0, "end": 0.6, "speaker": "A"}, {"start": 0.6, "end": 2, "speaker": "B"}])
    assert conf < 0.5 and flags


# ------------------------------------------------------------------------------- separation / mix / qa
@pytest.mark.parametrize("policy,expect", [(SeparationFallback.DUB_ONLY, "dub_only"), (SeparationFallback.FAIL, None)])
def test_separation_fallback_policy(video, settings, fake_engines, tmp_path, policy, expect):
    settings.separation_fallback = policy
    pipe = pl.DubbingPipeline(settings)
    if expect is None:
        with pytest.raises(pl.PipelineError):
            pipe.run(video, pl.JobOptions(mix_mode=MixMode.PRESERVE_BACKGROUND), tmp_path / "j", lambda *a: None)
        return
    result = pipe.run(video, pl.JobOptions(mix_mode=MixMode.PRESERVE_BACKGROUND), tmp_path / "j", lambda *a: None)
    report = json.loads((tmp_path / "j" / "job_report.json").read_text(encoding="utf-8"))
    assert report["mix_mode_used"] == expect and report["separation_fallback"]
    qa = json.loads(result.qa_report.read_text(encoding="utf-8"))
    assert qa["checks"]["separation_fallback"]["status"] == "REVIEW_REQUIRED" and result.qa_status == "REVIEW_REQUIRED"
    assert any("NOT removed" in w for w in result.warnings)


def test_dynamic_ducking_follower_and_gain_plan():
    sr = 1000
    f = DuckingFollower(sr, reduction_db=-6.0, attack_ms=20, release_ms=300)
    dialogue = np.zeros(3000, dtype=np.float32)
    dialogue[1000:2000] = 0.3
    g = f.process(dialogue)
    assert g[0] == pytest.approx(1.0) and g[1500] == pytest.approx(10 ** (-6 / 20), abs=0.01)
    assert 0.55 < g[2150] < 1.0 and g[2990] > 0.95, "release recovers the bed"
    g2 = f.process(np.zeros(500, dtype=np.float32))
    assert g2[0] >= g[-1], "state carries across blocks (no gain jump)"
    plan = plan_gains(-24.0, -14.0, target_lufs=-17.0, max_dialogue_db=6.0, margin_db=6.0, bed_enabled=True)
    assert plan.dialogue_db == 6.0 and plan.bed_db == pytest.approx(-10.0) and len(plan.notes) == 2
    assert plan_gains(None, None, -17.0, 6.0, 6.0, False).dialogue_db == 0.0


def test_qa_report_stems_and_release_gate(client, video, fake_engines):
    pid = client.post("/api/v1/projects", json={"name": "QA"}).json()["project_id"]
    job_id = _upload(client, pid, video)["job_id"]
    _wait(client, job_id, {"awaiting_review"})
    assert client.get(f"/api/v1/jobs/{job_id}/qa-report").status_code == 404
    assert client.post(f"/api/v1/jobs/{job_id}/release", json={"approved": True}).status_code == 409
    assert client.post(f"/api/v1/jobs/{job_id}/confirm", json={"base_version": 1}).status_code == 422
    voice = client.post("/api/v1/voice-library", json={
        "voice_name": "QA Voice", "engine": "parler",
        "voice_description_prompt": "A clear Hindi voice"}).json()
    assert _audition(client, voice["voice_id"])["status"] == "done"
    assert client.post(f"/api/v1/voice-library/{voice['voice_id']}/approval",
                       json={"approved": True}).json()["selectable"]
    draft = client.get(f"/api/v1/jobs/{job_id}/draft").json()
    patch = client.patch(f"/api/v1/jobs/{job_id}/draft", json={
        "base_version": draft["version"],
        "speakers": [{"speaker_id": speaker["speaker_id"], "voice_source": "cast", "voice_id": voice["voice_id"]}
                     for speaker in draft["speakers"]]})
    assert patch.status_code == 200, patch.text
    assert client.post(f"/api/v1/jobs/{job_id}/confirm",
                       json={"base_version": patch.json()["version"]}).status_code == 202
    job = _wait(client, job_id, {"succeeded", "failed"})
    assert job["state"] == "succeeded", job
    qa = client.get(f"/api/v1/jobs/{job_id}/qa-report").json()
    assert qa["status"] == job["qa_status"] and qa["checks"]["missing_audio_segments"]["status"] == "PASS"
    assert qa["checks"]["original_dialogue_leakage"]["status"] == "PASS"  # dub_only
    assert qa["loudness"]["dialogue"]["integrated_lufs"] is not None and "not_measured" in qa
    assert qa["mix"]["dynamic_ducking"] is True and qa["mix"]["target_lufs"] == -17.0
    for stem in ("master", "vocals", "dialogue", "bed", "mix"):
        assert client.get(f"/api/v1/jobs/{job_id}/stems/{stem}").status_code == 200, stem
    assert client.get(f"/api/v1/jobs/{job_id}/stems/background").status_code == 404  # no separation in this run
    r = client.post(f"/api/v1/jobs/{job_id}/release", json={"approved": True, "note": "listened, ok"})
    assert r.status_code == 200 and r.json()["release_approved"]
    jobs = client.app.state.jobs
    jobs._update(job_id, qa_status="FAILED", release_approved=False)
    assert client.post(f"/api/v1/jobs/{job_id}/release", json={"approved": True}).status_code == 409
    rep = client.get(f"/api/v1/jobs/{job_id}/report").json()
    assert rep["qa_status"] and rep["engines"] == ["parler"]


def test_original_plus_dub_reports_retained_original(video, settings, fake_engines, tmp_path):
    result = pl.DubbingPipeline(settings).run(video, pl.JobOptions(mix_mode=MixMode.ORIGINAL_PLUS_DUB), tmp_path / "j",
                                              lambda *a: None)
    qa = json.loads(result.qa_report.read_text(encoding="utf-8"))
    leak = qa["checks"]["original_dialogue_leakage"]
    assert leak["status"] == "WARNING" and "ducked by policy" in leak["detail"]
    assert 0.0 <= qa["metrics"]["original_dialogue_leakage_score"] <= 1.0


# ------------------------------------------------------------------------------- compatibility
def test_v2_records_still_load(settings):
    from backend.app.jobs import JobRecord
    from backend.services.projects import Project, ProjectVideo

    rec = JobRecord.model_validate({"job_id": "a" * 32, "state": "succeeded", "stage": "done", "progress": 1.0,
                                    "message": "", "input_filename": "x.mp4", "input_path": "x.mp4", "options": {},
                                    "created_at": "2026-10-01T00:00:00Z"})
    assert rec.qa_status is None and rec.release_approved is False and rec.engines == []
    p = Project.model_validate({"project_id": "b" * 32, "name": "old", "created_at": "2026-10-01T00:00:00Z",
                                "updated_at": "2026-10-01T00:00:00Z"})
    assert p.kind == "standalone"
    v = ProjectVideo.model_validate({"video_id": "c" * 32, "project_id": "b" * 32, "title": "t", "original_filename": "t.mp4",
                                     "stored_filename": "input.mp4", "size_bytes": 1, "duration": 1.0,
                                     "created_at": "2026-10-01T00:00:00Z", "updated_at": "2026-10-01T00:00:00Z"})
    assert v.episode_number is None
    draft = dr.DraftManifest.model_validate({
        "job_id": "a" * 32, "project_id": "b" * 32, "video_id": "c" * 32, "source_language": "en", "media_duration": 7.0,
        "created_at": "2026-10-01T00:00:00Z", "updated_at": "2026-10-01T00:00:00Z",
        "segments": [{"segment_id": "seg_00001", "start": 0.0, "end": 1.0, "original_text": "Hi", "translated_text": "नमस्ते",
                      "machine_translation": "नमस्ते", "speaker_id": "SPEAKER_00"}],
        "speakers": [{"speaker_id": "SPEAKER_00", "display_name": "Speaker 1"}]})
    spk = dr.refresh(draft, {}, settings).speakers[0]
    assert spk.engine is None and spk.mapping_status == "none" and spk.voice_source == VoiceSource.CLONE
    assert VoiceSource.CAST in spk.available_voice_options
    errors, _ = dr.validate_for_confirm(draft, set(), pl.DefaultVoiceBank(settings), settings)  # v2 call signature
    assert errors  # no clone reference -> same error as before
