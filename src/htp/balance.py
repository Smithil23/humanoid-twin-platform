"""Ankle-strategy balance controller.

The simplest genuine balance feedback for a standing humanoid, and the
same reflex people use on a moving bus: if the center of mass drifts
from the center of the support polygon, rotate the ankles to push it
back.

Every control tick the controller regulates the INSTANTANEOUS CAPTURE
POINT rather than the CoM itself:

    cp = com_xy + com_velocity_xy * sqrt(com_height / g)

The capture point is where the CoM *will settle* if nothing intervenes
(linear inverted pendulum model); steering it back to the support
center reacts to velocity before position error accumulates, which is
what makes push recovery robust:

    err = cp - support_polygon_centroid
    ankle_pitch_offset = +kp * err_x     (fore/aft)
    ankle_roll_offset  = -kp * err_y     (lateral)

Offsets are clamped and low-pass smoothed, applied equally to both
ankles on top of whatever the current motion commands. Sign conventions
match STAR1's URDF (verified empirically: positive ankle pitch tips the
body forward, positive ankle roll tips it left).

Limitations (deliberate, this is stage one of balance):
- assumes the robot faces roughly along +x (base yaw near zero)
- only acts while both feet have ground contact
- ankle strategy alone: recovers moderate pushes, not large ones
  (hip strategy and stepping are the later stages)
"""
from __future__ import annotations

import numpy as np

from .sim import Simulator


class BalanceController:
    def __init__(self, sim: Simulator,
                 kp: float = 2.8,
                 max_offset: float = 0.20, smooth: float = 0.35):
        self.sim = sim
        self.kp = kp
        self.max_offset = max_offset
        self.smooth = smooth            # low-pass factor per tick
        self._prev_com: np.ndarray | None = None
        self._off = np.zeros(2)         # [pitch, roll] current offsets

    def reset(self) -> None:
        self._prev_com = None
        self._off[:] = 0.0

    def update(self, dt: float,
               ref: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
        """Return (ankle_pitch_offset, ankle_roll_offset) in rad.

        ``ref`` is a CoM target offset [m] from the support-polygon
        centroid: (0, 0) means "stand centered"; a motion layer can
        command e.g. (0, +0.05) to deliberately shift weight left.
        The controller then steers the capture point to that spot
        instead of fighting the motion.
        """
        sim = self.sim
        # only act with real double support
        if len({c for c in range(sim.data.ncon)}) < 2:
            self._off *= 0.9
            return tuple(self._off)

        com3 = sim.data.subtree_com[1]
        com = com3[:2].copy()
        center = sim.support_polygon().mean(axis=0)
        if self._prev_com is None:
            vel = np.zeros(2)
        else:
            vel = (com - self._prev_com) / max(dt, 1e-6)
        self._prev_com = com

        # instantaneous capture point (LIPM): where the CoM will land
        omega = np.sqrt(max(com3[2], 0.3) / 9.81)
        cp = com + vel * omega
        err = cp - (center + np.asarray(ref))

        # signs measured on STAR1: +ankle_pitch pushes the CoM backward
        # (-x), +ankle_roll pushes it +y  ->  corrective mapping:
        raw_pitch = +self.kp * err[0]
        raw_roll = -self.kp * err[1]
        raw = np.clip([raw_pitch, raw_roll],
                      -self.max_offset, self.max_offset)
        self._off += self.smooth * (raw - self._off)
        return float(self._off[0]), float(self._off[1])

    ANKLE_PITCH = ("left_ankle_pitch_joint", "right_ankle_pitch_joint")
    ANKLE_ROLL = ("left_ankle_roll_joint", "right_ankle_roll_joint")

    def stance_sides(self, min_force: float = 30.0) -> tuple[str, ...]:
        """Which sides currently bear load ('left', 'right')."""
        ff = self.sim.foot_forces()
        out = tuple(
            ("left" if "left" in k else "right")
            for k, v in ff.items() if v >= min_force)
        return out or ("left", "right")
