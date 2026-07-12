"""Keyframe motion player: YAML motion scripts -> smooth joint targets.

A motion file describes poses at points in time; the player interpolates
between them with C1-smooth cosine easing, per joint, overlaid on a base
pose (normally the stand pose). Example:

    name: wave
    duration: 6.0
    loop: true
    keys:
      - t: 0.0
        pose: {right_shoulder_roll_joint: 0.0}
      - t: 1.0
        pose: {right_shoulder_roll_joint: -1.5}

Joints never mentioned stay at the base pose. A joint holds its first
keyed value before its first key and its last keyed value after its
last key (or wraps, if loop).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def _ease(u: float) -> float:
    """Cosine ease-in-out on [0, 1]."""
    return 0.5 * (1.0 - np.cos(np.pi * u))


class KeyframePlayer:
    def __init__(self, spec: dict, base: dict[str, float] | None = None):
        self.name: str = spec.get("name", "motion")
        self.duration: float = float(spec["duration"])
        self.loop: bool = bool(spec.get("loop", False))
        self.base = dict(base or {})
        # per-joint track: sorted list of (t, value)
        self.tracks: dict[str, list[tuple[float, float]]] = {}
        for key in sorted(spec["keys"], key=lambda k: k["t"]):
            t = float(key["t"])
            for joint, val in key["pose"].items():
                self.tracks.setdefault(joint, []).append((t, float(val)))

    @classmethod
    def from_file(cls, path: str | Path,
                  base: dict[str, float] | None = None) -> "KeyframePlayer":
        return cls(yaml.safe_load(Path(path).read_text()), base)

    # ---------------------------------------------------------------- api
    def finished(self, t: float) -> bool:
        return (not self.loop) and t >= self.duration

    def progress(self, t: float) -> float:
        if self.loop:
            return (t % self.duration) / self.duration
        return min(t / self.duration, 1.0)

    def targets(self, t: float) -> dict[str, float]:
        """Joint targets at time t, overlaid on the base pose."""
        if self.loop:
            t = t % self.duration
        else:
            t = min(t, self.duration)
        out = dict(self.base)
        for joint, track in self.tracks.items():
            out[joint] = self._sample(track, t)
        return out

    def _sample(self, track: list[tuple[float, float]], t: float) -> float:
        if t <= track[0][0]:
            if self.loop and len(track) > 1:
                # wrap: blend from last key (at duration) back to first
                t0, v0 = track[-1][0] - self.duration, track[-1][1]
                t1, v1 = track[0]
                if t1 > t0:
                    u = (t - t0) / (t1 - t0)
                    return v0 + (v1 - v0) * _ease(np.clip(u, 0, 1))
            return track[0][1]
        for i in range(len(track) - 1):
            t0, v0 = track[i]
            t1, v1 = track[i + 1]
            if t0 <= t <= t1:
                u = (t - t0) / max(t1 - t0, 1e-9)
                return v0 + (v1 - v0) * _ease(u)
        if self.loop and len(track) >= 1:
            # after last key: blend toward first key at t = duration
            t0, v0 = track[-1]
            t1, v1 = track[0][0] + self.duration, track[0][1]
            if t1 > t0:
                u = (t - t0) / (t1 - t0)
                return v0 + (v1 - v0) * _ease(np.clip(u, 0, 1))
        return track[-1][1]


def list_motions(folder: str | Path = "configs/motions") -> dict[str, Path]:
    """Available motion files, name -> path."""
    folder = Path(folder)
    out = {}
    if folder.is_dir():
        for p in sorted(folder.glob("*.yaml")):
            out[p.stem] = p
    return out
