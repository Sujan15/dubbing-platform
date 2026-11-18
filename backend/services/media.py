# hindi-dubbing-platform/backend/services/media.py
"""FFmpeg / FFprobe / WAV helpers.

Every external process is launched with an argument *list* (never a shell string), every
path is passed through ``os.fspath`` so Windows paths with spaces, backslashes or
non-ASCII characters reach FFmpeg unchanged, and every output is written to a temporary
sibling first and moved into place atomically.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import soundfile as sf

log = logging.getLogger("dubbing.media")

PathLike = Union[str, os.PathLike]
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    fps: float
    width: int
    height: int
    video_codec: str
    audio_streams: int
    audio_sample_rate: int
    audio_channels: int


class FFmpeg:
    def __init__(self, ffmpeg_bin: str, ffprobe_bin: str, timeout_seconds: int):
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin
        self.timeout = timeout_seconds

    # ------------------------------------------------------------------ process
    def _run(self, args: Sequence[PathLike], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        cmd = [os.fspath(a) for a in args]
        log.debug("exec: %s", subprocess.list2cmdline(cmd))
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout or self.timeout,
                creationflags=_CREATE_NO_WINDOW,
                check=False,
            )
        except FileNotFoundError as exc:
            raise MediaError(f"Executable not found: {cmd[0]}. Install FFmpeg and add it to PATH.") from exc
        except subprocess.TimeoutExpired as exc:
            raise MediaError(f"{Path(cmd[0]).name} timed out after {exc.timeout}s") from exc
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
            tail = "\n".join(stderr[-15:])
            raise MediaError(f"{Path(cmd[0]).name} failed (exit {proc.returncode}):\n{tail}")
        return proc

    def version(self) -> str:
        out = self._run([self.ffmpeg_bin, "-hide_banner", "-version"], timeout=30).stdout
        first = out.decode("utf-8", errors="replace").splitlines()
        self._run([self.ffprobe_bin, "-hide_banner", "-version"], timeout=30)
        return first[0] if first else "unknown"

    # ------------------------------------------------------------------ probing
    def probe(self, path: PathLike) -> MediaInfo:
        proc = self._run(
            [self.ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
            timeout=120,
        )
        data = json.loads(proc.stdout.decode("utf-8", errors="replace") or "{}")
        streams = data.get("streams", [])
        video = next(
            (s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")),
            None,
        )
        audio = [s for s in streams if s.get("codec_type") == "audio"]
        if video is None:
            raise MediaError("Input has no video stream.")
        if not audio:
            raise MediaError("Input has no audio stream; there is no dialogue to dub.")

        duration = _to_float(data.get("format", {}).get("duration")) or _to_float(video.get("duration"))
        if not duration or duration <= 0:
            raise MediaError("Could not determine media duration.")
        fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate")) or 0.0
        return MediaInfo(
            duration=duration,
            fps=round(fps, 3),
            width=int(video.get("width") or 0),
            height=int(video.get("height") or 0),
            video_codec=str(video.get("codec_name") or "unknown"),
            audio_streams=len(audio),
            audio_sample_rate=int(audio[0].get("sample_rate") or 0),
            audio_channels=int(audio[0].get("channels") or 0),
        )

    def measure_loudness(self, path: PathLike) -> dict:
        """EBU R128 integrated loudness (LUFS), loudness range (LU) and true peak (dBTP) of a file."""
        proc = subprocess.run(
            [self.ffmpeg_bin, "-hide_banner", "-nostdin", "-nostats", "-i", os.fspath(path),
             "-af", "ebur128=peak=true", "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout, creationflags=_CREATE_NO_WINDOW,
            check=False)
        text = proc.stderr.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise MediaError(f"ffmpeg ebur128 failed (exit {proc.returncode}): {text[-400:]}")
        summary = text[text.rfind("Summary:"):] if "Summary:" in text else text
        out = {"integrated_lufs": _grab(summary, r"I:\s+(-?[\d.]+|-inf)\s+LUFS"),
               "loudness_range_lu": _grab(summary, r"LRA:\s+(-?[\d.]+)\s+LU"),
               "true_peak_dbtp": _grab(summary, r"Peak:\s+(-?[\d.]+|-inf)\s+dBFS")}
        return out

    def audio_duration(self, path: PathLike) -> float:
        info = sf.info(os.fspath(path))
        return info.frames / float(info.samplerate)

    # ------------------------------------------------------------------ transforms
    def extract_audio(self, src: PathLike, dst: Path, sample_rate: int, channels: int, codec: str = "pcm_s16le") -> Path:
        tmp = _tmp_sibling(dst)
        try:
            self._run([
                self.ffmpeg_bin, "-hide_banner", "-nostdin", "-y", "-v", "error",
                "-i", src, "-map", "0:a:0", "-vn", "-sn", "-dn",
                "-ac", str(channels), "-ar", str(sample_rate), "-c:a", codec, tmp,
            ])
            os.replace(tmp, dst)
        finally:
            _unlink(tmp)
        return dst

    def resample(self, src: PathLike, dst: Path, sample_rate: int, channels: int, codec: str = "pcm_s16le") -> Path:
        return self.extract_audio(src, dst, sample_rate, channels, codec)

    def atempo(self, src: PathLike, dst: Path, ratio: float) -> Path:
        """Pitch-preserving speed-up. ``ratio`` > 1 shortens the clip by that factor."""
        if not 0.5 <= ratio <= 100.0:
            raise MediaError(f"atempo ratio out of range: {ratio}")
        tmp = _tmp_sibling(dst)
        try:
            self._run([
                self.ffmpeg_bin, "-hide_banner", "-nostdin", "-y", "-v", "error",
                "-i", src, "-filter:a", f"atempo={ratio:.6f}", "-c:a", "pcm_f32le", tmp,
            ])
            os.replace(tmp, dst)
        finally:
            _unlink(tmp)
        return dst

    def mux(self, video: PathLike, audio: PathLike, dst: Path, bitrate: str) -> Path:
        ext = dst.suffix.lower()
        codec = {".webm": "libopus", ".avi": "libmp3lame"}.get(ext, "aac")
        args: list[PathLike] = [
            self.ffmpeg_bin, "-hide_banner", "-nostdin", "-y", "-v", "error",
            "-i", video, "-i", audio,
            "-map", "0:v:0", "-map", "1:a:0",
        ]
        if ext == ".mkv":
            args += ["-map", "0:s?", "-c:s", "copy"]
        args += [
            "-map_metadata", "0", "-map_chapters", "0",
            "-c:v", "copy",
            "-af", "alimiter=limit=0.891:attack=5:release=50:level=0",
            "-c:a", codec, "-b:a", bitrate,
            "-metadata:s:a:0", "language=hin", "-metadata:s:a:0", "title=Hindi (AI dub)",
        ]
        if ext in {".mp4", ".mov", ".m4v"}:
            args += ["-movflags", "+faststart"]
        tmp = _tmp_sibling(dst)
        args.append(tmp)
        try:
            self._run(args)
            os.replace(tmp, dst)
        finally:
            _unlink(tmp)
        return dst


# ---------------------------------------------------------------------- WAV I/O
def read_audio(path: PathLike, frames: int = -1, start: int = 0) -> tuple[np.ndarray, int]:
    """Read float32 audio shaped (samples, channels)."""
    data, sr = sf.read(os.fspath(path), frames=frames, start=start, dtype="float32", always_2d=True)
    return data, sr


def write_audio(path: Path, data: np.ndarray, sample_rate: int, subtype: str = "FLOAT") -> Path:
    tmp = _tmp_sibling(path)
    try:
        sf.write(os.fspath(tmp), data, sample_rate, subtype=subtype, format="WAV")
        os.replace(tmp, path)
    finally:
        _unlink(tmp)
    return path


def trim_silence(mono: np.ndarray, sample_rate: int, threshold_db: float = -40.0, pad_ms: float = 30.0) -> np.ndarray:
    """Trim leading/trailing silence relative to the clip peak (TTS engines pad generously)."""
    if mono.size == 0:
        return mono
    peak = float(np.max(np.abs(mono)))
    if peak <= 1e-6:
        return mono[:0]
    win = max(1, int(sample_rate * 0.01))
    n = mono.size // win
    if n == 0:
        return mono
    frames = mono[: n * win].reshape(n, win)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    active = np.where(20 * np.log10(rms / peak) > threshold_db)[0]
    if active.size == 0:
        return mono[:0]
    pad = int(sample_rate * pad_ms / 1000)
    start = max(0, active[0] * win - pad)
    end = min(mono.size, (active[-1] + 1) * win + pad)
    return mono[start:end]


def rms_db(mono: np.ndarray) -> float:
    if mono.size == 0:
        return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(np.square(mono, dtype=np.float64))) + 1e-9))


# ---------------------------------------------------------------------- helpers
def _tmp_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f"{path.stem}.part-{uuid.uuid4().hex[:8]}{path.suffix}")


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("Could not remove temporary file %s", path)


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _grab(text: str, pattern: str) -> Optional[float]:
    import re
    m = re.search(pattern, text)
    if not m:
        return None
    v = m.group(1)
    return -120.0 if v == "-inf" else float(v)


def _parse_rate(value: Optional[str]) -> float:
    if not value or value in {"0/0", "0"}:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0
