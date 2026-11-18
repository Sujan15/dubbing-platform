# hindi-dubbing-platform/backend/services/audio_qa.py

"""Automated quality gate for a generated episode.

A finished TTS job is not a finished dub. ``build_qa_report`` measures what can be measured with
signal processing and FFmpeg and turns it into one of four statuses:

    PASS             nothing to look at
    WARNING          minor, release allowed
    REVIEW_REQUIRED  a human must listen before release (leakage, fallback mix, overflow lines ...)
    FAILED           release blocked (missing lines, clipping, leakage above the fail threshold ...)

Metrics (all 0..1, higher = worse unless stated):

* ``original_dialogue_leakage_score`` - speech-band energy of the *bed* inside Hindi dialogue
  windows relative to the original vocals there. preserve_background: residual speech left in the
  background stem by the separator; original_plus_dub: the ducked original by policy; dub_only: 0.
* ``separation_quality_score`` (higher = better) - 1 - leakage, penalised when the vocals stem
  carries energy outside dialogue (music bled into vocals) or either stem is empty.
* ``background_preservation_score`` (higher = better) - bed energy kept outside dialogue windows
  relative to the original track there (how much music/SFX survived).

These are calibrated heuristics for *triage*, not proof of quality. The report says so, and the
release decision stays with a listener. Every number that needs a model (ASR re-transcription,
embedding similarity) is left as an explicit ``not_measured`` entry rather than faked.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from backend.app.config import MixMode, Settings
from backend.services.media import FFmpeg, MediaError, read_audio

log = logging.getLogger("dubbing.qa")

SPEECH_BAND = (300.0, 3400.0)
STATUS_ORDER = ["PASS", "WARNING", "REVIEW_REQUIRED", "FAILED"]


def _worse(a: str, b: str) -> str:
    return a if STATUS_ORDER.index(a) >= STATUS_ORDER.index(b) else b


def speech_band_energy(mono: np.ndarray, sr: int) -> float:
    if mono.size < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(mono.astype(np.float64)))
    freqs = np.fft.rfftfreq(mono.size, 1.0 / sr)
    band = (freqs >= SPEECH_BAND[0]) & (freqs <= SPEECH_BAND[1])
    return float(np.sum(spec[band] ** 2) / mono.size)


def _windows_energy(path: Path, windows: list[tuple[float, float]], inside: bool, max_seconds: float = 600.0) -> float:
    """Mean speech-band energy inside (or outside) the given windows, sampled at most ``max_seconds``."""
    info = sf.info(os.fspath(path))
    sr, total = info.samplerate, info.frames
    spans = []
    if inside:
        spans = [(int(s * sr), int(e * sr)) for s, e in windows]
    else:
        cursor = 0
        for s, e in sorted(windows):
            a, b = int(s * sr), int(e * sr)
            if a > cursor:
                spans.append((cursor, a))
            cursor = max(cursor, b)
        if cursor < total:
            spans.append((cursor, total))
    budget = int(max_seconds * sr)
    energies, used = [], 0
    for a, b in spans:
        a, b = max(0, a), min(total, b)
        if b - a < int(0.1 * sr):
            continue
        take = min(b - a, budget - used)
        if take <= 0:
            break
        mono = read_audio(path, frames=take, start=a)[0].mean(axis=1)
        energies.append(speech_band_energy(mono, sr))
        used += take
    return float(np.mean(energies)) if energies else 0.0


def clipping_stats(path: Path, threshold: float = 0.999) -> dict:
    info = sf.info(os.fspath(path))
    sr, total = info.samplerate, info.frames
    clipped, peak = 0, 0.0
    block = 60 * sr
    for start in range(0, total, block):
        data = read_audio(path, frames=min(block, total - start), start=start)[0]
        a = np.abs(data)
        clipped += int((a >= threshold).sum())
        peak = max(peak, float(a.max()) if a.size else 0.0)
    return {"clipped_samples": clipped, "peak": round(peak, 4), "ratio": clipped / max(total * info.channels, 1)}


def build_qa_report(*, settings: Settings, ff: FFmpeg, mode_used: MixMode, mode_requested: MixMode,
                    segments: list[dict], timing: dict, mix_dir: Path, master: Path, vocals: Path,
                    background: Optional[Path], output: Path, separation_fallback: bool,
                    voice_resolution: dict, mix_report: dict) -> dict:
    status = "PASS"
    checks: dict[str, dict] = {}

    def add(name: str, st: str, value=None, detail: str = "") -> None:
        nonlocal status
        checks[name] = {"status": st, "value": value, "detail": detail}
        status = _worse(status, st)

    windows = [(float(s["start_time"]), float(s["start_time"]) + float(s.get("generated_duration") or 0.0))
               for s in segments]
    dialogue_stem = mix_dir / "dialogue.wav"
    bed_stem = mix_dir / "bed.wav"

    # ---- missing / silent lines ---------------------------------------------------
    missing = [s["segment_id"] for s in segments
               if not s.get("generated_audio_path") or not Path(s["generated_audio_path"]).is_file()]
    silent = []
    for s in segments:
        p = s.get("generated_audio_path")
        if p and Path(p).is_file():
            data = read_audio(p)[0]
            if data.size == 0 or float(np.max(np.abs(data))) < 1e-4:
                silent.append(s["segment_id"])
    add("missing_audio_segments", "FAILED" if missing or silent else "PASS", len(missing) + len(silent),
        f"missing={missing[:10]} silent={silent[:10]}" if missing or silent else "every line has audio")

    # ---- loudness -----------------------------------------------------------------
    loud: dict[str, dict] = {}
    for name, path in (("dialogue", dialogue_stem), ("bed", bed_stem), ("mix", mix_dir / "mix.wav"), ("output", output)):
        if path.is_file():
            try:
                loud[name] = ff.measure_loudness(path)
            except MediaError as exc:
                loud[name] = {"error": str(exc)[:200]}
    d_lufs = (loud.get("dialogue") or {}).get("integrated_lufs")
    if d_lufs is not None:
        off = abs(d_lufs - settings.dialogue_target_lufs)
        add("dialogue_loudness", "PASS" if off <= 2.0 else ("WARNING" if off <= 4.0 else "REVIEW_REQUIRED"),
            d_lufs, f"target {settings.dialogue_target_lufs} LUFS")
    b_lufs = (loud.get("bed") or {}).get("integrated_lufs")
    if d_lufs is not None and b_lufs is not None and b_lufs > -100:
        margin = d_lufs - b_lufs
        add("dialogue_bed_margin", "PASS" if margin >= settings.min_dialogue_bed_margin_db else
            ("WARNING" if margin >= settings.min_dialogue_bed_margin_db - 3 else "REVIEW_REQUIRED"),
            round(margin, 2), f"dialogue minus bed integrated loudness (min {settings.min_dialogue_bed_margin_db:g} dB, "
                              "static; dynamic ducking adds more under speech)")
    tp = (loud.get("output") or {}).get("true_peak_dbtp")
    if tp is not None:
        add("true_peak", "PASS" if tp <= settings.true_peak_ceiling_dbtp + 0.1 else "FAILED", tp,
            f"ceiling {settings.true_peak_ceiling_dbtp} dBTP")

    # ---- clipping -----------------------------------------------------------------
    clip = clipping_stats(mix_dir / "mix.wav") if (mix_dir / "mix.wav").is_file() else {"clipped_samples": 0, "ratio": 0}
    add("clipping", "PASS" if clip["clipped_samples"] == 0 else "FAILED",
        clip["clipped_samples"], f"peak {clip.get('peak')}")

    # ---- leakage / separation ---------------------------------------------------
    metrics: dict = {"original_dialogue_leakage_score": 0.0, "separation_quality_score": None,
                     "background_preservation_score": None}
    if mode_used == MixMode.DUB_ONLY:
        add("original_dialogue_leakage", "PASS", 0.0, "dub_only: the original soundtrack is not in the mix")
    elif windows and bed_stem.is_file():
        bed_in = _windows_energy(bed_stem, windows, inside=True)
        voc_in = _windows_energy(vocals, windows, inside=True)
        leak = float(np.clip(bed_in / max(voc_in, 1e-12), 0.0, 1.0)) if voc_in > 0 else 0.0
        metrics["original_dialogue_leakage_score"] = round(leak, 4)
        if mode_used == MixMode.ORIGINAL_PLUS_DUB:
            detail = ("original_plus_dub keeps the original dialogue, ducked by policy "
                      f"(DUCK_DB={settings.duck_db:g}); residual speech-band ratio {leak:.2f}")
            add("original_dialogue_leakage", "REVIEW_REQUIRED" if separation_fallback else "WARNING", round(leak, 4), detail)
        else:
            st = "FAILED" if leak >= settings.leakage_fail_threshold else (
                "REVIEW_REQUIRED" if leak >= settings.leakage_review_threshold else "PASS")
            add("original_dialogue_leakage", st, round(leak, 4),
                "speech-band energy left in the separated background under Hindi lines, relative to the original vocals")
        if background is not None:
            voc_out = _windows_energy(vocals, windows, inside=False)
            music_bleed = float(np.clip(voc_out / max(voc_in, 1e-12), 0.0, 1.0)) if voc_in > 0 else 0.0
            empty = []
            for name, p in (("vocals", vocals), ("background", background)):
                try:
                    if (ff.measure_loudness(p).get("integrated_lufs") or -120) < -60:
                        empty.append(name)
                except MediaError:
                    pass
            quality = max(0.0, (1.0 - leak) * (1.0 - 0.5 * music_bleed)) * (0.0 if empty else 1.0)
            metrics["separation_quality_score"] = round(quality, 4)
            add("separation_quality", "FAILED" if empty else ("REVIEW_REQUIRED" if quality < 0.5 else
                ("WARNING" if quality < 0.75 else "PASS")), round(quality, 4),
                f"music bleed into vocals {music_bleed:.2f}" + (f"; empty stems: {empty}" if empty else ""))
            bg_out = _windows_energy(bed_stem, windows, inside=False)
            master_out = _windows_energy(master, windows, inside=False)
            pres = float(np.clip(bg_out / max(master_out, 1e-12), 0.0, 1.0)) if master_out > 0 else 0.0
            metrics["background_preservation_score"] = round(pres, 4)
            add("background_preservation", "WARNING" if pres < 0.2 else "PASS", round(pres, 4),
                "bed energy outside dialogue relative to the original track")
    if separation_fallback:
        add("separation_fallback", "REVIEW_REQUIRED", mode_used.value,
            f"{mode_requested.value} was requested but no separated stems existed; policy SEPARATION_FALLBACK="
            f"{settings.separation_fallback.value} applied - the original dialogue was NOT removed")

    # ---- timing -----------------------------------------------------------------
    summary = timing.get("summary", {})
    timing_segments = timing.get("segments", [])
    overflow_ids = {
        r.get("segment_id", i) for i, r in enumerate(timing_segments)
        if r.get("status") == "atempo_capped_overflow" or r.get("overlaps_next")
    }
    overflow = len(overflow_ids) if timing_segments else summary.get("atempo_capped_overflow", 0)
    overlaps = sum(1 for r in timing_segments if r.get("overlaps_next"))
    add("timing_overflow", "FAILED" if overflow else "PASS", overflow,
        f"lines still exceed their dialogue slot after timing adjustment; {overlaps} line(s) run into the next line")
    fast = sum(1 for r in timing.get("segments", []) if float(r.get("speed_adjustment", 1.0)) > 1.2)
    if fast:
        add("aggressive_speedup", "WARNING", fast, "lines sped up by more than 1.2x; check naturalness")

    # ---- voice consistency / substitutions ------------------------------------------
    fallbacks = [spk for spk, rv in voice_resolution.items() if rv.get("fallback_applied")]
    add("voice_substitutions", "REVIEW_REQUIRED" if fallbacks else "PASS", len(fallbacks),
        f"speakers that did not get the requested voice: {fallbacks}" if fallbacks else "every speaker got the requested voice")
    engines = sorted({rv.get("engine") for rv in voice_resolution.values() if rv.get("engine")})

    not_measured = {
        "speaker_attribution_accuracy": "needs a labelled set; see /speakers suggestions + review",
        "pronunciation_intelligibility": "ASR re-transcription not run (enable in a later release)",
        "character_identity_consistency": "embedding comparison of generated lines vs approved reference not run",
        "translation_adequacy": "human review gate",
    }
    report = {"status": status, "checks": checks, "metrics": metrics, "loudness": loud, "clipping": clip,
              "mix": mix_report, "engines": engines, "not_measured": not_measured,
              "release_blocked": status == "FAILED",
              "note": "Signal-based triage; a listener makes the release decision."}
    return report


def write_report(path: Path, report: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
