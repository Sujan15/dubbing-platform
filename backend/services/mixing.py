# hindi-dubbing-platform/backend/services/mixing.py
"""Dialogue-aware mixing.

    Hindi dialogue stem  ->  activity envelope (frame RMS)  ->  sidechain gain with attack / release
    bed (separated background | ducked original | nothing)  ->  static margin gain  ->  dynamic gain
    mix = bed * gain + dialogue * dialogue_gain

* ``DuckingFollower`` is stateful so the mix can be rendered block by block (60 s) without gain
  discontinuities: the release tail carries over into the next block.
* ``plan_gains`` turns measured loudness (EBU R128 via FFmpeg) into two static gains: push the
  dialogue stem towards ``DIALOGUE_TARGET_LUFS`` (capped by ``DIALOGUE_NORMALIZE_MAX_DB``) and keep
  the bed at least ``MIN_DIALOGUE_BED_MARGIN_DB`` below it. The final mix applies uniform peak
  protection before muxing; the QA report measures sample clipping and true peak.
* The old static envelope (``pipeline.ducking_envelope``) remains for the legacy path and tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_MS = 10.0


@dataclass
class MixGains:
    dialogue_db: float
    bed_db: float
    dialogue_lufs_before: float | None
    bed_lufs_before: float | None
    notes: list[str]


def plan_gains(dialogue_lufs: float | None, bed_lufs: float | None, target_lufs: float, max_dialogue_db: float,
               margin_db: float, bed_enabled: bool) -> MixGains:
    notes: list[str] = []
    dialogue_db = 0.0
    if dialogue_lufs is not None and dialogue_lufs > -100:
        dialogue_db = float(np.clip(target_lufs - dialogue_lufs, -max_dialogue_db, max_dialogue_db))
        if abs(target_lufs - dialogue_lufs) > max_dialogue_db:
            notes.append(f"dialogue needed {target_lufs - dialogue_lufs:+.1f} dB to reach {target_lufs} LUFS; "
                         f"capped at {dialogue_db:+.1f} dB (DIALOGUE_NORMALIZE_MAX_DB)")
    bed_db = 0.0
    if bed_enabled and bed_lufs is not None and bed_lufs > -100 and dialogue_lufs is not None and dialogue_lufs > -100:
        dialogue_after = dialogue_lufs + dialogue_db
        ceiling = dialogue_after - margin_db
        if bed_lufs > ceiling:
            bed_db = float(ceiling - bed_lufs)
            notes.append(f"bed ({bed_lufs:.1f} LUFS) louder than dialogue - {margin_db:g} dB; lowered {bed_db:+.1f} dB")
    return MixGains(round(dialogue_db, 2), round(bed_db, 2), dialogue_lufs, bed_lufs, notes)


class DuckingFollower:
    """Sample-accurate gain curve for the bed, driven by dialogue activity.

    ``reduction_db`` is the gain applied while dialogue is active (e.g. -6 dB for music under Hindi,
    DUCK_DB for the original track). Attack = how fast the bed drops when dialogue starts,
    release = how fast it recovers. Both are one-pole smoothers at frame rate.
    """

    def __init__(self, sample_rate: int, reduction_db: float, attack_ms: float, release_ms: float,
                 threshold_db: float = -45.0):
        self.sr = sample_rate
        self.hop = max(1, int(sample_rate * FRAME_MS / 1000))
        self.floor = 10 ** (reduction_db / 20)
        self.threshold = 10 ** (threshold_db / 20)
        self.a_att = float(np.exp(-FRAME_MS / max(attack_ms, 1e-3)))
        self.a_rel = float(np.exp(-FRAME_MS / max(release_ms, 1e-3)))
        self.state = 1.0

    def process(self, dialogue: np.ndarray) -> np.ndarray:
        n = dialogue.size
        if n == 0:
            return np.ones(0, dtype=np.float32)
        frames = int(np.ceil(n / self.hop))
        padded = np.zeros(frames * self.hop, dtype=np.float32)
        padded[:n] = dialogue
        rms = np.sqrt(np.mean(padded.reshape(frames, self.hop) ** 2, axis=1) + 1e-12)
        target = np.where(rms > self.threshold, self.floor, 1.0).astype(np.float64)
        gains = np.empty(frames, dtype=np.float32)
        g = self.state
        for i in range(frames):
            t = target[i]
            coef = self.a_att if t < g else self.a_rel
            g = coef * g + (1.0 - coef) * t
            gains[i] = g
        self.state = g
        return np.repeat(gains, self.hop)[:n]


def apply_gain_db(x: np.ndarray, db: float) -> np.ndarray:
    return x if abs(db) < 1e-6 else x * (10 ** (db / 20))
