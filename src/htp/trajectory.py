"""Trajectory generators: smooth, cyclic joint-space references."""
from __future__ import annotations

import numpy as np


def cosine_ramp(t: float, period: float, amplitude: float) -> float:
    """0 -> amplitude -> 0 over one period, C1-smooth (zero end velocity)."""
    if t <= 0.0 or t >= period:
        return 0.0
    return amplitude * 0.5 * (1.0 - np.cos(2.0 * np.pi * t / period))


def squat_targets(t: float, period: float = 4.0, depth: float = 0.5,
                  start: float = 2.0,
                  base: dict[str, float] | None = None) -> dict[str, float]:
    """Symmetric squat: knees bend, hip/ankle pitch keep the torso level.

    ``depth`` is the peak knee angle [rad]; hip and ankle compensate with
    -depth/2 each so the trunk stays vertical and the COM travels straight
    down - the standard first dynamic validation of a contact model.
    ``base`` is the stance pose the squat is overlaid on (STAR1's upright
    stance has hip +30 / knee -60 / ankle +30 deg; positive knee motion
    bends the leg, so the squat drives knee toward zero and beyond).
    """
    if base is None:
        base = {}
    s = cosine_ramp(t - start, period, depth)
    out: dict[str, float] = {}
    for side in ("left", "right"):
        out[f"{side}_knee_joint"] = base.get(f"{side}_knee_joint", 0.0) + s
        out[f"{side}_hip_pitch_joint"] = (
            base.get(f"{side}_hip_pitch_joint", 0.0) - s / 2.0
        )
        out[f"{side}_ankle_pitch_joint"] = (
            base.get(f"{side}_ankle_pitch_joint", 0.0) - s / 2.0
        )
    return out
