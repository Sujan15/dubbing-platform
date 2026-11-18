
# hindi-dubbing-platform/tests/test_isolated_tts.py


"""Unit tests for the isolated TTS worker protocol and lifecycle.

All model engines and subprocesses are faked. No model download or inference
is performed by this test module.
"""

from __future__ import annotations

import io
import json
import queue
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from backend.app.config import TTSEngine
from backend.services.tts import isolated
from backend.services.tts.base import EngineMetadata, TTSEngineError, VoiceSpec


class FakeSettings:
    def apply_runtime_environment(self):
        pass


class FakeInnerEngine:
    def __init__(self, engine):
        self.engine = engine
        self.sample_rate = 24000
        self.fingerprint = {"model": "fake-model"}
        self.unloaded = False
        self.last_synthesis = None

    def get_model_metadata(self):
        return EngineMetadata(
            engine=self.engine,
            model_id="fake-model",
            model_revision="test",
            sample_rate=self.sample_rate,
            device="cpu",
        )

    def synthesize(self, text, spec, seed, speed=1.0):
        self.last_synthesis = {
            "text": text,
            "spec": spec,
            "seed": seed,
            "speed": speed,
        }
        return np.ones(2400, dtype=np.float32) * 0.1

    def unload(self):
        self.unloaded = True


class FakePipe:
    """Minimal file-like object compatible with Popen pipe usage."""

    def __init__(self):
        self.closed = False
        self.written = []

    def write(self, value):
        if self.closed:
            raise ValueError("I/O operation on closed pipe")
        self.written.append(value)
        return len(value)

    def flush(self):
        if self.closed:
            raise ValueError("I/O operation on closed pipe")

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.stdin = FakePipe()
        self.stdout = FakePipe()
        self.stderr = FakePipe()
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def make_proxy(engine=TTSEngine.PARLER):
    """Create an IsolatedEngine without launching a real subprocess."""

    proxy = object.__new__(isolated.IsolatedEngine)
    proxy.engine = engine
    proxy._timeout = 60
    proxy._lock = threading.Lock()
    proxy._stderr_lock = threading.Lock()
    proxy._stderr_tail = deque(maxlen=isolated._STDERR_TAIL_SIZE)
    proxy._stdout_queue = queue.Queue()
    proxy._proc = FakeProcess()
    return proxy


def parse_protocol(output):
    responses = []

    for line in output.getvalue().splitlines():
        if line.startswith(isolated._PROTOCOL_PREFIX):
            payload = line[len(isolated._PROTOCOL_PREFIX):]
            responses.append(json.loads(payload))

    return responses


def test_worker_environment_forwards_hugging_face_tokens(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "legacy-test-token")

    env = isolated._worker_environment()

    assert env["HF_TOKEN"] == "hf-test-token"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "legacy-test-token"


@pytest.fixture
def fake_worker_environment(monkeypatch):
    import backend.app.config as config
    import backend.services.tts as tts_package

    instances = []

    def fake_engine_class(engine):
        def factory(settings, device):
            instance = FakeInnerEngine(engine)
            instances.append(instance)
            return instance

        return factory

    monkeypatch.setattr(config, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(tts_package, "engine_class", fake_engine_class)

    return instances


# ---------------------------------------------------------------------------
# Parent-side protocol tests
# ---------------------------------------------------------------------------

def test_call_accepts_valid_prefixed_protocol_response():
    proxy = make_proxy()

    proxy._stdout_queue.put(
        isolated._PROTOCOL_PREFIX
        + json.dumps({"ok": True, "sample_rate": 24000})
        + "\n"
    )

    reply = proxy._call({"op": "metadata"})

    assert reply["ok"] is True
    assert reply["sample_rate"] == 24000
    assert proxy._proc.stdin.written


def test_call_ignores_non_protocol_stdout():
    proxy = make_proxy()

    proxy._stdout_queue.put("third-party startup message\n")
    proxy._stdout_queue.put(
        isolated._PROTOCOL_PREFIX
        + json.dumps({"ok": True, "value": "ready"})
        + "\n"
    )

    reply = proxy._call({"op": "metadata"})

    assert reply["value"] == "ready"


def test_call_rejects_worker_error_response():
    proxy = make_proxy()

    proxy._stdout_queue.put(
        isolated._PROTOCOL_PREFIX
        + json.dumps({
            "ok": False,
            "error": "model initialization failed",
        })
        + "\n"
    )

    with pytest.raises(TTSEngineError, match="model initialization failed"):
        proxy._call({"op": "metadata"})


def test_call_rejects_malformed_protocol_json():
    proxy = make_proxy()

    proxy._stdout_queue.put(
        isolated._PROTOCOL_PREFIX + "{invalid-json}\n"
    )

    with pytest.raises(TTSEngineError, match="malformed protocol JSON"):
        proxy._call({"op": "metadata"})

    assert proxy._proc.terminated


def test_call_rejects_non_object_protocol_response():
    proxy = make_proxy()

    proxy._stdout_queue.put(
        isolated._PROTOCOL_PREFIX
        + json.dumps(["not", "an", "object"])
        + "\n"
    )

    with pytest.raises(TTSEngineError, match="non-object"):
        proxy._call({"op": "metadata"})

    assert proxy._proc.terminated


def test_call_rejects_unexpected_worker_exit():
    proxy = make_proxy()
    proxy._proc.returncode = 17

    with pytest.raises(TTSEngineError, match="worker exited"):
        proxy._call({"op": "metadata"})


def test_call_times_out_and_terminates_worker():
    proxy = make_proxy()

    with pytest.raises(TTSEngineError, match="timed out"):
        proxy._call({"op": "metadata"}, timeout=1)

    assert proxy._proc.terminated


# ---------------------------------------------------------------------------
# Child-side serve() tests
# ---------------------------------------------------------------------------

def test_serve_metadata_and_graceful_quit(
    monkeypatch,
    fake_worker_environment,
):
    monkeypatch.setattr(
        isolated.sys,
        "stdin",
        io.StringIO('{"op":"metadata"}\n{"op":"quit"}\n'),
    )

    captured_stdout = io.StringIO()
    monkeypatch.setattr(isolated.sys, "stdout", captured_stdout)

    result = isolated.serve(
        engine=TTSEngine.PARLER,
        device="cpu",
        data_dir=None,
    )

    responses = parse_protocol(captured_stdout)

    assert result == 0
    assert len(responses) == 2

    metadata = responses[0]
    assert metadata["ok"] is True
    assert metadata["sample_rate"] == 24000
    assert metadata["metadata"]["model_id"] == "fake-model"
    assert metadata["fingerprint"]["model"] == "fake-model"

    assert responses[1] == {"ok": True}
    assert fake_worker_environment[0].unloaded is True


def test_serve_synthesis_handoff_and_quit(
    monkeypatch,
    fake_worker_environment,
    tmp_path,
):
    output_wav = tmp_path / "worker-output.wav"

    spec = VoiceSpec(
        engine=TTSEngine.PARLER,
        voice_prompt="A clear professional Hindi voice",
    )

    request_spec = {
        **{
            key: value
            for key, value in spec.__dict__.items()
            if key != "engine"
        },
        "engine": spec.engine.value,
        "reference_path": None,
    }

    request = {
        "op": "synthesize",
        "text": "नमस्ते",
        "spec": request_spec,
        "seed": 42,
        "speed": 1.0,
        "out": str(output_wav),
    }

    monkeypatch.setattr(
        isolated.sys,
        "stdin",
        io.StringIO(
            json.dumps(request, ensure_ascii=False)
            + "\n"
            + '{"op":"quit"}\n'
        ),
    )

    captured_stdout = io.StringIO()
    monkeypatch.setattr(isolated.sys, "stdout", captured_stdout)

    written = {}

    def fake_write_audio(path, audio, sample_rate):
        written["path"] = Path(path)
        written["audio"] = audio
        written["sample_rate"] = sample_rate
        Path(path).write_bytes(b"fake-wav")

    monkeypatch.setattr(isolated, "write_audio", fake_write_audio)

    result = isolated.serve(
        engine=TTSEngine.PARLER,
        device="cpu",
        data_dir=str(tmp_path),
    )

    responses = parse_protocol(captured_stdout)
    inner = fake_worker_environment[0]

    assert result == 0
    assert responses[0]["ok"] is True
    assert responses[0]["sample_rate"] == 24000
    assert responses[0]["path"] == str(output_wav)

    assert written["path"] == output_wav
    assert written["sample_rate"] == 24000
    assert written["audio"].dtype == np.float32

    assert inner.last_synthesis["text"] == "नमस्ते"
    assert inner.last_synthesis["seed"] == 42
    assert inner.unloaded is True


def test_serve_unknown_operation_returns_error_and_continues(
    monkeypatch,
    fake_worker_environment,
):
    monkeypatch.setattr(
        isolated.sys,
        "stdin",
        io.StringIO('{"op":"unknown-operation"}\n{"op":"quit"}\n'),
    )

    captured_stdout = io.StringIO()
    monkeypatch.setattr(isolated.sys, "stdout", captured_stdout)

    result = isolated.serve(
        engine=TTSEngine.PARLER,
        device="cpu",
        data_dir=None,
    )

    responses = parse_protocol(captured_stdout)

    assert result == 0
    assert responses[0]["ok"] is False
    assert "unknown op" in responses[0]["error"]
    assert responses[1]["ok"] is True
    assert fake_worker_environment[0].unloaded is True


def test_serve_reports_startup_failure(monkeypatch):
    import backend.services.tts as tts_package

    def failing_engine_class(engine):
        def factory(settings, device):
            raise RuntimeError("fake model startup failure")

        return factory

    monkeypatch.setattr(
        tts_package,
        "engine_class",
        failing_engine_class,
    )

    captured_stdout = io.StringIO()
    monkeypatch.setattr(isolated.sys, "stdout", captured_stdout)

    result = isolated.serve(
        engine=TTSEngine.PARLER,
        device="cpu",
        data_dir=None,
    )

    responses = parse_protocol(captured_stdout)

    assert result == 1
    assert len(responses) == 1
    assert responses[0]["ok"] is False
    assert "fake model startup failure" in responses[0]["error"]


# ---------------------------------------------------------------------------
# Parent-side synthesis validation tests
# ---------------------------------------------------------------------------

def test_synthesize_returns_valid_float32_mono_audio(
    monkeypatch,
    tmp_path,
):
    proxy = make_proxy()
    proxy.sample_rate = 24000
    proxy._scratch = tmp_path

    def fake_call(request):
        Path(request["out"]).write_bytes(b"fake-wav")
        return {"ok": True, "sample_rate": 24000}

    proxy._call = fake_call

    expected = np.ones((2400, 1), dtype=np.float32) * 0.25

    monkeypatch.setattr(
        isolated,
        "read_audio",
        lambda path: (expected, 24000),
    )

    audio = proxy.synthesize(
        text="नमस्ते",
        spec=VoiceSpec(engine=TTSEngine.PARLER),
        seed=7,
    )

    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.shape == (2400,)
    assert np.allclose(audio, 0.25)
    assert list(tmp_path.glob("*.wav")) == []


def test_synthesize_rejects_wrong_response_sample_rate(
    monkeypatch,
    tmp_path,
):
    proxy = make_proxy()
    proxy.sample_rate = 24000
    proxy._scratch = tmp_path

    proxy._call = lambda request: {
        "ok": True,
        "sample_rate": 22050,
    }

    with pytest.raises(TTSEngineError, match="expected 24000 Hz"):
        proxy.synthesize(
            "नमस्ते",
            VoiceSpec(engine=TTSEngine.PARLER),
            seed=1,
        )


def test_synthesize_rejects_non_finite_audio(
    monkeypatch,
    tmp_path,
):
    proxy = make_proxy()
    proxy.sample_rate = 24000
    proxy._scratch = tmp_path

    def fake_call(request):
        # The worker must create the file before read_audio is called.
        Path(request["out"]).write_bytes(b"fake-wav")
        return {
            "ok": True,
            "sample_rate": 24000,
        }

    proxy._call = fake_call

    invalid_audio = np.ones((100, 1), dtype=np.float32)
    invalid_audio[0, 0] = np.nan

    monkeypatch.setattr(
        isolated,
        "read_audio",
        lambda path: (invalid_audio, 24000),
    )

    with pytest.raises(TTSEngineError, match="non-finite"):
        proxy.synthesize(
            "नमस्ते",
            VoiceSpec(engine=TTSEngine.PARLER),
            seed=1,
        )


def test_synthesize_rejects_missing_output_file(tmp_path):
    proxy = make_proxy()
    proxy.sample_rate = 24000
    proxy._scratch = tmp_path

    proxy._call = lambda request: {
        "ok": True,
        "sample_rate": 24000,
    }

    with pytest.raises(TTSEngineError, match="did not create"):
        proxy.synthesize(
            "नमस्ते",
            VoiceSpec(engine=TTSEngine.PARLER),
            seed=1,
        )


# ---------------------------------------------------------------------------
# Process shutdown tests
# ---------------------------------------------------------------------------

def test_unload_uses_graceful_shutdown(tmp_path):
    proxy = make_proxy()
    proxy._scratch = tmp_path

    temporary_file = tmp_path / "temporary.wav"
    temporary_file.write_bytes(b"temporary")

    def successful_quit(request, timeout=None):
        assert request == {"op": "quit"}
        assert timeout == isolated._DEFAULT_SHUTDOWN_TIMEOUT
        return {"ok": True}

    proxy._call = successful_quit

    proxy.unload()

    assert proxy._proc.returncode == 0
    assert proxy._proc.terminated is False
    assert not temporary_file.exists()


def test_unload_falls_back_to_process_termination(tmp_path):
    proxy = make_proxy()
    proxy._scratch = tmp_path

    def failed_quit(request, timeout=None):
        raise TTSEngineError("simulated graceful shutdown failure")

    proxy._call = failed_quit

    proxy.unload()

    assert proxy._proc.terminated is True
    assert proxy._proc.stdin.closed
    assert proxy._proc.stdout.closed
    assert proxy._proc.stderr.closed


def test_unload_without_process_still_cleans_scratch(tmp_path):
    proxy = make_proxy()
    proxy._proc = None
    proxy._scratch = tmp_path

    temporary_file = tmp_path / "temporary.wav"
    temporary_file.write_bytes(b"temporary")

    proxy.unload()

    assert not temporary_file.exists()
