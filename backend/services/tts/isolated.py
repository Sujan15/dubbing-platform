# # hindi-dubbing-platform/backend/services/tts/isolated.py

# """Run a TTS adapter in a separate Python process (optionally a different venv).

# Why: (1) the Parler and IndicF5 stacks may need different ``transformers`` versions, (2) a long-lived
# server fragments model memory after Parler/IndicF5 jobs - a child process returns every byte
# when it exits, (3) a crashing model library must not take the API down.

# Protocol (one JSON object per line on stdin/stdout, binary audio exchanged as WAV files on disk):

#     -> {"op": "synthesize", "text": ..., "spec": {...}, "seed": 7, "speed": 1.0, "out": "<path>.wav"}
#     <- {"ok": true, "sample_rate": 24000, "path": "<path>.wav"}
#     -> {"op": "metadata"}   <- {"ok": true, "metadata": {...}, "fingerprint": {...}, "sample_rate": ...}
#     -> {"op": "quit"}

# The parent only ever reads what the child prints; a malformed or missing line is an error, never
# an instruction. ``IsolatedEngine`` exposes the same ``BaseTTSEngine`` surface as an in-process
# adapter, so the pipeline does not know the difference.
# """

# from __future__ import annotations

# import json
# import logging
# import os
# import subprocess
# import sys
# import threading
# import uuid
# from dataclasses import asdict
# from pathlib import Path
# from typing import Optional

# import numpy as np

# from backend.app.config import REPO_ROOT, Settings, TTSEngine
# from backend.services.media import read_audio, write_audio
# from backend.services.tts.base import BaseTTSEngine, EngineMetadata, TTSEngineError, VoiceSpec

# log = logging.getLogger("dubbing.tts.isolated")
# _CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


# class IsolatedEngine(BaseTTSEngine):
#     """Proxy around ``python -m backend.services.tts.worker --engine <name>`` in another interpreter."""

#     def __init__(self, settings: Settings, device: str, engine: TTSEngine, python: str, inner_cls: type[BaseTTSEngine]):
#         super().__init__(settings, device)
#         self.engine = engine
#         self.capabilities = inner_cls.capabilities
#         self._python = python
#         self._lock = threading.Lock()
#         self._scratch = settings.work_dir / "isolated_tts" / uuid.uuid4().hex[:8]
#         self._scratch.mkdir(parents=True, exist_ok=True)
#         env = dict(os.environ)
#         env.setdefault("PYTHONIOENCODING", "utf-8")
#         env["PYTHONPATH"] = os.fspath(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
#         self._proc = subprocess.Popen(
#             [python, "-m", "backend.services.tts.worker", "--engine", engine.value, "--device", device,
#              "--data-dir", os.fspath(settings.data_dir)],
#             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
#             cwd=os.fspath(REPO_ROOT), env=env, creationflags=_CREATE_NO_WINDOW)
#         self._stderr_tail: list[str] = []
#         threading.Thread(target=self._pump_stderr, daemon=True).start()
#         meta = self._call({"op": "metadata"})
#         self.sample_rate = int(meta["sample_rate"])
#         self.fingerprint = dict(meta["fingerprint"])
#         self._metadata = EngineMetadata(**{**meta["metadata"], "engine": engine, "isolated": True})

#     def _pump_stderr(self) -> None:
#         assert self._proc.stderr is not None
#         for line in self._proc.stderr:
#             line = line.rstrip()
#             self._stderr_tail = (self._stderr_tail + [line])[-40:]
#             log.debug("[%s worker] %s", self.engine.value, line)

#     def _call(self, request: dict) -> dict:
#         with self._lock:
#             if self._proc.poll() is not None:
#                 raise TTSEngineError(f"{self.engine.value} worker exited (code {self._proc.returncode}): "
#                                      + "\n".join(self._stderr_tail[-10:]))
#             assert self._proc.stdin and self._proc.stdout
#             self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
#             self._proc.stdin.flush()
#             line = self._proc.stdout.readline()
#         if not line:
#             raise TTSEngineError(f"{self.engine.value} worker produced no reply: " + "\n".join(self._stderr_tail[-10:]))
#         try:
#             reply = json.loads(line)
#         except ValueError as exc:
#             raise TTSEngineError(f"{self.engine.value} worker replied with non-JSON: {line[:200]!r}") from exc
#         if not isinstance(reply, dict) or not reply.get("ok"):
#             raise TTSEngineError(f"{self.engine.value} worker error: {str(reply.get('error') if isinstance(reply, dict) else reply)[:500]}")
#         return reply

#     def get_model_metadata(self) -> EngineMetadata:
#         return self._metadata

#     def synthesize(self, text: str, spec: VoiceSpec, seed: int, speed: float = 1.0) -> np.ndarray:
#         out = self._scratch / f"{uuid.uuid4().hex[:12]}.wav"
#         payload = asdict(spec)
#         payload["engine"] = spec.engine.value
#         payload["reference_path"] = os.fspath(spec.reference_path) if spec.reference_path else None
#         try:
#             reply = self._call({"op": "synthesize", "text": text, "spec": payload, "seed": seed, "speed": speed,
#                                 "out": os.fspath(out)})
#             data, sr = read_audio(Path(reply["path"]))
#             if sr != self.sample_rate:
#                 raise TTSEngineError(f"worker returned {sr} Hz, expected {self.sample_rate} Hz")
#             return data[:, 0].astype(np.float32)
#         finally:
#             out.unlink(missing_ok=True)

#     def unload(self) -> None:
#         try:
#             if self._proc.poll() is None:
#                 try:
#                     self._call({"op": "quit"})
#                 except TTSEngineError:
#                     pass
#                 self._proc.wait(timeout=30)
#         except Exception:  # the process is being discarded anyway
#             self._proc.kill()
#         finally:
#             for f in self._scratch.glob("*"):
#                 f.unlink(missing_ok=True)
#             try:
#                 self._scratch.rmdir()
#             except OSError:
#                 pass


# def serve(engine: TTSEngine, device: str, data_dir: Optional[str]) -> int:
#     """Child-process main loop (see ``backend.services.tts.worker``)."""
#     from backend.app.config import get_settings
#     from backend.services.tts import engine_class

#     if data_dir:
#         os.environ["DATA_DIR"] = data_dir
#     settings = get_settings()
#     settings.apply_runtime_environment()
#     inner = engine_class(engine)(settings, device)
#     stdin, stdout = sys.stdin, sys.stdout
#     for raw in stdin:
#         try:
#             req = json.loads(raw)
#             op = req.get("op")
#             if op == "quit":
#                 print(json.dumps({"ok": True}), file=stdout, flush=True)
#                 break
#             if op == "metadata":
#                 md = asdict(inner.get_model_metadata())
#                 md["engine"] = inner.engine.value
#                 print(json.dumps({"ok": True, "metadata": md, "fingerprint": inner.fingerprint,
#                                   "sample_rate": inner.sample_rate}), file=stdout, flush=True)
#                 continue
#             if op == "synthesize":
#                 sp = dict(req["spec"])
#                 sp["engine"] = TTSEngine(sp["engine"])
#                 sp["reference_path"] = Path(sp["reference_path"]) if sp.get("reference_path") else None
#                 spec = VoiceSpec(**sp)
#                 wav = inner.synthesize(req["text"], spec, int(req["seed"]), float(req.get("speed", 1.0)))
#                 write_audio(Path(req["out"]), wav.astype(np.float32), inner.sample_rate)
#                 print(json.dumps({"ok": True, "sample_rate": inner.sample_rate, "path": req["out"]}), file=stdout, flush=True)
#                 continue
#             print(json.dumps({"ok": False, "error": f"unknown op {op!r}"}), file=stdout, flush=True)
#         except Exception as exc:  # report every failure to the parent instead of dying silently
#             print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), file=stdout, flush=True)
#     inner.unload()
#     return 0





"""Run a TTS adapter in a separate Python process (optionally a different venv).

Why:
1. Parler and IndicF5 may require different Transformers versions.
2. A child process releases its model memory when it exits.
3. A model-library crash should not take down the API process.

Protocol:
- One prefixed JSON object per line on stdout.
- Binary audio is exchanged through WAV files on disk.
- Non-protocol stdout output is ignored by the parent.
- Diagnostic output is captured from stderr.

The parent exposes the same BaseTTSEngine surface as an in-process adapter.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import uuid
import time

from collections import deque
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from backend.app.config import REPO_ROOT, Settings, TTSEngine
# from backend.services.media import read_audio
from backend.services.media import read_audio, write_audio
from backend.services.tts.base import (
    BaseTTSEngine,
    EngineMetadata,
    TTSEngineError,
    VoiceSpec,
)

log = logging.getLogger("dubbing.tts.isolated")

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

_PROTOCOL_PREFIX = "__DUBBING_TTS__:"

_STDERR_TAIL_SIZE = 40
_ERROR_TAIL_SIZE = 10

_DEFAULT_SHUTDOWN_TIMEOUT = 10
_PROCESS_TERMINATE_TIMEOUT = 5
_PROCESS_FINAL_WAIT_TIMEOUT = 5


def _worker_environment() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


class IsolatedEngine(BaseTTSEngine):
    """Proxy around an isolated TTS worker process."""

    def __init__(
        self,
        settings: Settings,
        device: str,
        engine: TTSEngine,
        python: str,
        inner_cls: type[BaseTTSEngine],
    ):
        super().__init__(settings, device)

        self.engine = engine
        self.capabilities = inner_cls.capabilities
        self._python = python

        self._lock = threading.Lock()
        self._stderr_lock = threading.Lock()

        self._timeout = max(
            60,
            int(settings.tts_isolated_worker_timeout),
        )

        self._stdout_queue: queue.Queue[Optional[str]] = queue.Queue()

        self._stderr_tail: deque[str] = deque(
            maxlen=_STDERR_TAIL_SIZE
        )

        self._scratch = (
            settings.work_dir
            / "isolated_tts"
            / uuid.uuid4().hex[:8]
        )

        self._scratch.mkdir(
            parents=True,
            exist_ok=False,
        )

        self._proc: Optional[subprocess.Popen] = None

        self.sample_rate: int
        self.fingerprint: dict
        self._metadata: EngineMetadata

        env = _worker_environment()

        env.setdefault(
            "PYTHONIOENCODING",
            "utf-8",
        )

        env["PYTHONUNBUFFERED"] = "1"

        existing_pythonpath = env.get("PYTHONPATH", "")

        env["PYTHONPATH"] = (
            os.fspath(REPO_ROOT)
            + (
                os.pathsep + existing_pythonpath
                if existing_pythonpath
                else ""
            )
        )

        try:
            self._proc = subprocess.Popen(
                [
                    python,
                    "-m",
                    "backend.services.tts.worker",
                    "--engine",
                    engine.value,
                    "--device",
                    device,
                    "--data-dir",
                    os.fspath(settings.data_dir),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=os.fspath(REPO_ROOT),
                env=env,
                creationflags=_CREATE_NO_WINDOW,
            )

            self._stdout_thread = threading.Thread(
                target=self._pump_stdout,
                daemon=True,
                name=f"{engine.value}-stdout",
            )

            self._stderr_thread = threading.Thread(
                target=self._pump_stderr,
                daemon=True,
                name=f"{engine.value}-stderr",
            )

            self._stdout_thread.start()
            self._stderr_thread.start()

            meta = self._call(
                {"op": "metadata"}
            )

            self.sample_rate = int(
                meta["sample_rate"]
            )

            if self.sample_rate <= 0:
                raise TTSEngineError(
                    f"{engine.value} worker returned an invalid sample rate"
                )

            self.fingerprint = dict(
                meta["fingerprint"]
            )

            self._metadata = EngineMetadata(
                **{
                    **meta["metadata"],
                    "engine": engine,
                    "isolated": True,
                }
            )

        except Exception:
            self._terminate_worker()
            self._cleanup_scratch()
            raise

    def _stderr_snapshot(self) -> list[str]:
        """Return a thread-safe copy of recent worker diagnostics."""

        with self._stderr_lock:
            return list(self._stderr_tail)

    def _pump_stdout(self) -> None:
        """Read worker stdout continuously for timeout-safe communication."""

        proc = self._proc

        if proc is None or proc.stdout is None:
            self._stdout_queue.put(None)
            return

        try:
            for line in proc.stdout:
                self._stdout_queue.put(line)

        except (OSError, ValueError) as exc:
            log.debug(
                "[%s worker] stdout reader stopped: %s",
                self.engine.value,
                exc,
            )

        finally:
            self._stdout_queue.put(None)

    def _pump_stderr(self) -> None:
        """Capture bounded worker diagnostics without blocking the child."""

        proc = self._proc

        if proc is None or proc.stderr is None:
            return

        try:
            for line in proc.stderr:
                line = line.rstrip()

                with self._stderr_lock:
                    self._stderr_tail.append(line)

                log.debug(
                    "[%s worker] %s",
                    self.engine.value,
                    line,
                )

        except (OSError, ValueError) as exc:
            log.debug(
                "[%s worker] stderr reader stopped: %s",
                self.engine.value,
                exc,
            )

    def _terminate_worker(self) -> None:
        """Terminate or kill a failed worker and close its pipes."""

        proc = self._proc

        if proc is None:
            return

        try:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(
                        timeout=_PROCESS_TERMINATE_TIMEOUT
                    )

                except subprocess.TimeoutExpired:
                    log.warning(
                        "%s worker did not terminate; killing process",
                        self.engine.value,
                    )

                    proc.kill()

                    try:
                        proc.wait(
                            timeout=_PROCESS_FINAL_WAIT_TIMEOUT
                        )
                    except subprocess.TimeoutExpired:
                        log.error(
                            "%s worker did not exit after kill",
                            self.engine.value,
                        )

                except OSError:
                    log.exception(
                        "Could not terminate %s worker",
                        self.engine.value,
                    )

                    try:
                        proc.kill()
                        proc.wait(
                            timeout=_PROCESS_FINAL_WAIT_TIMEOUT
                        )
                    except (OSError, subprocess.TimeoutExpired):
                        pass

        finally:
            for pipe in (
                proc.stdin,
                proc.stdout,
                proc.stderr,
            ):
                if pipe is not None:
                    try:
                        pipe.close()
                    except (OSError, ValueError):
                        pass

    def _cleanup_scratch(self) -> None:
        """Remove temporary WAV files and the worker scratch directory."""

        try:
            for file_path in self._scratch.glob("*"):
                try:
                    file_path.unlink(
                        missing_ok=True
                    )
                except OSError:
                    log.warning(
                        "Could not remove isolated TTS file: %s",
                        file_path,
                    )

            try:
                self._scratch.rmdir()
            except OSError:
                pass

        except OSError:
            log.exception(
                "Could not clean isolated TTS scratch directory: %s",
                self._scratch,
            )

    def _call(
        self,
        request: dict,
        timeout: Optional[int] = None,
    ) -> dict:
        """Send one request and wait for its matching protocol response.

        The timeout applies to the complete response operation, not to each
        non-protocol stdout line.
        """

        proc = self._proc

        if proc is None:
            raise TTSEngineError(
                f"{self.engine.value} worker was not initialized"
            )

        wait_timeout = (
            self._timeout
            if timeout is None
            else max(1, int(timeout))
        )

        with self._lock:
            if proc.poll() is not None:
                raise TTSEngineError(
                    f"{self.engine.value} worker exited "
                    f"(code {proc.returncode}):\n"
                    + "\n".join(
                        self._stderr_snapshot()[-_ERROR_TAIL_SIZE:]
                    )
                )

            if proc.stdin is None:
                raise TTSEngineError(
                    f"{self.engine.value} worker stdin is unavailable"
                )

            try:
                serialized = json.dumps(
                    request,
                    ensure_ascii=False,
                    allow_nan=False,
                )

                proc.stdin.write(
                    serialized + "\n"
                )

                proc.stdin.flush()

            except (
                BrokenPipeError,
                OSError,
                ValueError,
                TypeError,
            ) as exc:
                self._terminate_worker()

                raise TTSEngineError(
                    f"{self.engine.value} worker input pipe failed: {exc}\n"
                    + "\n".join(
                        self._stderr_snapshot()[-_ERROR_TAIL_SIZE:]
                    )
                ) from exc

            # deadline = (
            #     __import__("time").monotonic()
            #     + wait_timeout
            # )
            deadline = time.monotonic() + wait_timeout

            while True:
                # remaining = (
                #     deadline
                #     - __import__("time").monotonic()
                # )
                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    self._terminate_worker()

                    raise TTSEngineError(
                        f"{self.engine.value} worker timed out after "
                        f"{wait_timeout} seconds while handling "
                        f"{request.get('op')!r}"
                    )

                try:
                    line = self._stdout_queue.get(
                        timeout=remaining
                    )

                except queue.Empty as exc:
                    self._terminate_worker()

                    raise TTSEngineError(
                        f"{self.engine.value} worker timed out after "
                        f"{wait_timeout} seconds while handling "
                        f"{request.get('op')!r}"
                    ) from exc

                if line is None:
                    raise TTSEngineError(
                        f"{self.engine.value} worker closed stdout "
                        f"(exit code: {proc.poll()}):\n"
                        + "\n".join(
                            self._stderr_snapshot()[-_ERROR_TAIL_SIZE:]
                        )
                    )

                line = line.rstrip("\r\n")

                if not line.startswith(_PROTOCOL_PREFIX):
                    log.debug(
                        "[%s worker] ignored non-protocol stdout: %s",
                        self.engine.value,
                        line[:300],
                    )
                    continue

                payload = line[len(_PROTOCOL_PREFIX):]

                try:
                    reply = json.loads(payload)

                except (ValueError, TypeError) as exc:
                    self._terminate_worker()

                    raise TTSEngineError(
                        f"{self.engine.value} worker returned malformed "
                        f"protocol JSON: {payload[:200]!r}"
                    ) from exc

                if not isinstance(reply, dict):
                    self._terminate_worker()

                    raise TTSEngineError(
                        f"{self.engine.value} worker returned "
                        "a non-object protocol response"
                    )

                if not reply.get("ok"):
                    error = reply.get(
                        "error",
                        "Unknown worker error",
                    )

                    raise TTSEngineError(
                        f"{self.engine.value} worker error: "
                        f"{str(error)[:500]}"
                    )

                return reply

    def get_model_metadata(self) -> EngineMetadata:
        return self._metadata

    def synthesize(
        self,
        text: str,
        spec: VoiceSpec,
        seed: int,
        speed: float = 1.0,
    ) -> np.ndarray:
        """Synthesize audio through the isolated worker."""

        if not text or not text.strip():
            raise TTSEngineError(
                "Cannot synthesize empty text"
            )

        if self._proc is None or self._proc.poll() is not None:
            raise TTSEngineError(
                f"{self.engine.value} worker is not running"
            )

        out = (
            self._scratch
            / f"{uuid.uuid4().hex[:12]}.wav"
        )

        payload = asdict(spec)

        payload["engine"] = spec.engine.value

        payload["reference_path"] = (
            os.fspath(spec.reference_path)
            if spec.reference_path
            else None
        )

        try:
            reply = self._call(
                {
                    "op": "synthesize",
                    "text": text,
                    "spec": payload,
                    "seed": seed,
                    "speed": speed,
                    "out": os.fspath(out),
                }
            )

            response_sample_rate = int(
                reply["sample_rate"]
            )

            if response_sample_rate != self.sample_rate:
                raise TTSEngineError(
                    f"{self.engine.value} worker response declared "
                    f"{response_sample_rate} Hz; expected "
                    f"{self.sample_rate} Hz"
                )

            if not out.is_file():
                raise TTSEngineError(
                    f"{self.engine.value} worker did not create "
                    f"the expected WAV file"
                )

            data, sample_rate = read_audio(out)

            if sample_rate != self.sample_rate:
                raise TTSEngineError(
                    f"{self.engine.value} worker returned "
                    f"{sample_rate} Hz; expected "
                    f"{self.sample_rate} Hz"
                )

            if data.ndim != 2 or data.shape[1] < 1:
                raise TTSEngineError(
                    f"{self.engine.value} worker returned invalid audio shape: "
                    f"{data.shape}"
                )

            audio = data[:, 0].astype(
                np.float32,
                copy=False,
            )

            if audio.size == 0:
                raise TTSEngineError(
                    f"{self.engine.value} worker returned empty audio"
                )

            if not np.isfinite(audio).all():
                raise TTSEngineError(
                    f"{self.engine.value} worker returned non-finite audio"
                )

            return audio

        finally:
            try:
                out.unlink(
                    missing_ok=True
                )
            except OSError:
                log.warning(
                    "Could not remove temporary worker WAV: %s",
                    out,
                )

    def unload(self) -> None:
        """Gracefully unload the model and stop the isolated process."""

        proc = self._proc

        if proc is None:
            self._cleanup_scratch()
            return

        try:
            if proc.poll() is None:
                try:
                    self._call(
                        {"op": "quit"},
                        timeout=_DEFAULT_SHUTDOWN_TIMEOUT,
                    )

                    proc.wait(
                        timeout=30
                    )

                except (
                    TTSEngineError,
                    subprocess.TimeoutExpired,
                    OSError,
                ):
                    log.warning(
                        "%s worker graceful shutdown failed; "
                        "terminating process",
                        self.engine.value,
                    )

                    self._terminate_worker()

        finally:
            if proc.poll() is None:
                self._terminate_worker()

            else:
                # Close remaining handles even if the child exited normally.
                for pipe in (
                    proc.stdin,
                    proc.stdout,
                    proc.stderr,
                ):
                    if pipe is not None:
                        try:
                            pipe.close()
                        except (OSError, ValueError):
                            pass

            self._cleanup_scratch()


def serve(
    engine: TTSEngine,
    device: str,
    data_dir: Optional[str],
) -> int:
    """Child-process main loop with protected JSON protocol stdout."""

    protocol_stdout = sys.stdout
    inner = None

    def emit(payload: dict) -> None:
        """Write one prefixed protocol response."""

        protocol_stdout.write(
            _PROTOCOL_PREFIX
            + json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )

        protocol_stdout.flush()

    try:
        # Keep all Python-level third-party print() calls away from protocol stdout.
        with redirect_stdout(sys.stderr):
            from backend.app.config import get_settings
            from backend.services.tts import engine_class

            if data_dir:
                os.environ["DATA_DIR"] = data_dir

            settings = get_settings()

            settings.apply_runtime_environment()

            inner = engine_class(engine)(
                settings,
                device,
            )

    except Exception as exc:
        log.exception(
            "Isolated %s worker startup failed",
            engine.value,
        )

        try:
            emit(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        except (OSError, ValueError):
            pass

        return 1

    try:
        for raw in sys.stdin:
            try:
                req = json.loads(raw)

                if not isinstance(req, dict):
                    raise ValueError(
                        "Request must be a JSON object"
                    )

                op = req.get("op")

                if op == "quit":
                    emit(
                        {"ok": True}
                    )
                    break

                if op == "metadata":
                    with redirect_stdout(sys.stderr):
                        md = asdict(
                            inner.get_model_metadata()
                        )

                    md["engine"] = inner.engine.value

                    emit(
                        {
                            "ok": True,
                            "metadata": md,
                            "fingerprint": inner.fingerprint,
                            "sample_rate": inner.sample_rate,
                        }
                    )

                    continue

                if op == "synthesize":
                    sp = dict(
                        req["spec"]
                    )

                    sp["engine"] = TTSEngine(
                        sp["engine"]
                    )

                    sp["reference_path"] = (
                        Path(sp["reference_path"])
                        if sp.get("reference_path")
                        else None
                    )

                    spec = VoiceSpec(**sp)

                    with redirect_stdout(sys.stderr):
                        wav = inner.synthesize(
                            req["text"],
                            spec,
                            int(req["seed"]),
                            float(req.get("speed", 1.0)),
                        )

                        write_audio(
                            Path(req["out"]),
                            wav.astype(np.float32),
                            inner.sample_rate,
                        )

                    emit(
                        {
                            "ok": True,
                            "sample_rate": inner.sample_rate,
                            "path": req["out"],
                        }
                    )

                    continue

                emit(
                    {
                        "ok": False,
                        "error": f"unknown op {op!r}",
                    }
                )

            except Exception as exc:
                log.exception(
                    "Isolated %s worker request failed",
                    engine.value,
                )

                emit(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        return 0

    finally:
        if inner is not None:
            try:
                with redirect_stdout(sys.stderr):
                    inner.unload()

            except Exception:
                log.exception(
                    "Isolated %s worker unload failed",
                    engine.value,
                )
