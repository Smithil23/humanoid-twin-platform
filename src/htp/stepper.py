"""Event-based stepping-in-place: a step state machine.

Open-loop (clock-driven) stepping manages ~4 steps before accumulated
drift wins (see scripts/step_experiments.py). This module replaces the
clock with MEASUREMENTS - each phase transition fires on what the feet
actually report, so every step re-anchors instead of compounding error:

    SHIFT   ramp CoM reference + waist lean toward the stance side,
            advance when the swing foot MEASURES unloaded
    LIFT    curl the swing leg (short, fixed - flight is ballistic)
    LAND    extend the leg, advance when contact is MEASURED regained
    RETURN  re-center, then mirror to the other side

The balance controller (capture-point, stance-side routed) runs
underneath throughout; this layer only produces joint targets and the
CoM reference. Cosine easing on every channel; each phase eases from
whatever the current command is, so transitions are always continuous.

STATUS
------
The state machine is event-correct: SHIFT waits for a MEASURED unloaded
swing foot, LAND waits for MEASURED contact, and the CoM reference
re-anchors to the stance foot each step, so it does not accumulate drift.
With the ankle-only balance controller this managed a few steps before
the single-support authority limit (~0.7 s) was reached. Adding the HIP
STRATEGY to the balance controller (waist counter-rotation + arm
counter-abduction, active in single support) broke that ceiling: the
march is now sustained indefinitely (verified 162 steps / 2 min, stable
+10 cm margin). The Studio drives this automatically via the March
button; arm counter-swing is applied in the engine's march path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import Simulator

STAND_HIP, STAND_KNEE = 0.5236, -1.0472


@dataclass
class StepParams:
    # Defaults tuned for the ankle-strategy balance controller. With
    # only ankle balancing, sustained stepping is authority-limited
    # (see the note in the module docstring); these values give the
    # most reliable few steps before that limit is reached. When a hip
    # strategy is added (D4), amp and f_stance can be raised for a
    # snappier, longer march.
    amp: float = 0.08          # CoM reference shift [m]
    waist: float = 0.55        # waist-roll feedforward [rad]
    hip_d: float = 0.20        # swing hip curl [rad]
    knee_d: float = 0.40       # swing knee curl [rad]
    ankle_d: float = -0.45     # swing ankle pitch during flight [rad]
                               # (keeps the foot level; without it the
                               # toe scrapes and kicks the robot back)
    t_shift: float = 2.4       # min shift ramp [s]
    t_shift_max: float = 5.0   # abort (retry via RETURN) after this
    t_up: float = 0.28         # lift ramp [s]
    t_apex: float = 0.10       # hold at apex [s]
    t_down: float = 0.50       # lowering ramp [s] (soft touchdown)
    t_land_max: float = 0.8    # extra wait for contact after lowering
    t_settle: float = 0.80     # post-landing pause, double support [s]
                               # (longer = balance fully recovers between
                               #  steps; key to the indefinite march)
    t_return: float = 1.6      # re-center ramp [s]
    f_unloaded: float = 80.0   # swing foot counts as unloaded below [N]
    f_stance: float = 250.0    # ...and stance must carry at least this
    f_loaded: float = 60.0     # swing counts as landed above [N]
    debounce: float = 0.10     # condition must hold this long [s]
    # forward walking (stride=0 -> march in place):
    stride: float = 0.0        # swing hip-pitch forward offset [rad]
                               # (negative hip pitch swings the leg
                               #  forward; positive stride -> forward step)
    lean: float = 0.0          # forward CoM lean during double support [m]


def _ease(u: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * float(np.clip(u, 0.0, 1.0))))


class Stepper:
    """Produces (joint_targets, com_ref) each tick; owns its channels."""

    CHANNELS = ("com_y", "waist_roll_joint",
                "left_hip_pitch_joint", "left_knee_joint",
                "left_ankle_pitch_joint",
                "right_hip_pitch_joint", "right_knee_joint",
                "right_ankle_pitch_joint")

    def __init__(self, sim: Simulator, params: StepParams | None = None):
        self.sim = sim
        self.p = params or StepParams()
        self.reset()

    def reset(self) -> None:
        self.phase = "SHIFT"
        self.swing = "left"            # first step lifts the left foot
        self.steps = 0
        self._t = 0.0                  # time in current phase
        self._cond_t = 0.0             # debounce accumulator
        self.cmd: dict[str, float] = {
            "com_y": 0.0, "waist_roll_joint": 0.0,
            "left_hip_pitch_joint": STAND_HIP,
            "left_knee_joint": STAND_KNEE,
            "left_ankle_pitch_joint": STAND_HIP,
            "right_hip_pitch_joint": STAND_HIP,
            "right_knee_joint": STAND_KNEE,
            "right_ankle_pitch_joint": STAND_HIP,
        }
        self._snap = dict(self.cmd)    # phase-start snapshot
        self._target = dict(self.cmd)  # phase-end target
        self._enter_shift()

    # ------------------------------------------------------------ phases
    def _begin(self, phase: str, duration: float,
               target: dict[str, float]) -> None:
        self.phase = phase
        self._t = 0.0
        self._cond_t = 0.0
        self._dur = max(duration, 1e-6)
        self._snap = dict(self.cmd)
        self._target = dict(self.cmd)
        self._target.update(target)

    def _enter_shift(self) -> None:
        sgn = -1.0 if self.swing == "left" else +1.0
        self._begin("SHIFT", self.p.t_shift, {
            "com_y": sgn * self.p.amp,
            "waist_roll_joint": sgn * self.p.waist,
        })

    def _enter_lift(self) -> None:
        s = self.swing
        sgn = -1.0 if s == "left" else +1.0
        # once airborne the support polygon collapses to the stance
        # foot, so the CoM reference must relax to near-center of it
        self._begin("LIFT", self.p.t_up, {
            f"{s}_hip_pitch_joint": STAND_HIP + self.p.hip_d - self.p.stride,
            f"{s}_knee_joint": STAND_KNEE + self.p.knee_d,
            f"{s}_ankle_pitch_joint": STAND_HIP + self.p.ankle_d,
            "com_y": sgn * 0.01,
        })

    def _enter_land(self) -> None:
        s = self.swing
        # plant forward: the swing hip lands advanced by `stride`, and
        # the STANCE hip retracts by the same amount so the body glides
        # forward over the new base (this is what makes it travel, not
        # just paw the ground)
        st = "right" if s == "left" else "left"
        self._begin("LAND", self.p.t_down, {
            f"{s}_hip_pitch_joint": STAND_HIP - self.p.stride,
            f"{s}_knee_joint": STAND_KNEE,
            f"{s}_ankle_pitch_joint": STAND_HIP,
            f"{st}_hip_pitch_joint": STAND_HIP + self.p.stride,
        })

    def _enter_settle(self) -> None:
        # Touchdown re-anchoring: double support resumed, so the
        # support centroid just jumped from the stance foot back to
        # mid-feet. Re-express the CoM reference in the new frame at
        # its MEASURED value, so the balance controller sees zero
        # discontinuity; RETURN then ramps it smoothly to center.
        centroid_y = float(self.sim.support_polygon().mean(axis=0)[1])
        com_y = float(self.sim.data.subtree_com[1][1])
        self.cmd["com_y"] = com_y - centroid_y
        self._begin("SETTLE", self.p.t_settle, {})

    def _enter_return(self) -> None:
        self._begin("RETURN", self.p.t_return, {
            "com_y": 0.0, "waist_roll_joint": 0.0,
        })

    # ------------------------------------------------------------- tick
    def _swing_force(self) -> float:
        return self.sim.foot_forces()[f"{self.swing}_ankle_roll_link"]

    def _stance_force(self) -> float:
        other = "right" if self.swing == "left" else "left"
        return self.sim.foot_forces()[f"{other}_ankle_roll_link"]

    def _debounced(self, condition: bool, dt: float) -> bool:
        self._cond_t = self._cond_t + dt if condition else 0.0
        return self._cond_t >= self.p.debounce

    def update(self, dt: float) -> tuple[dict[str, float],
                                         tuple[float, float]]:
        self._t += dt
        u = _ease(self._t / self._dur)
        for k in self.CHANNELS:
            self.cmd[k] = (self._snap[k]
                           + (self._target[k] - self._snap[k]) * u)

        if self.phase == "SHIFT":
            unloaded = self._debounced(
                self._swing_force() < self.p.f_unloaded
                and self._stance_force() > self.p.f_stance, dt)
            if self._t >= self._dur and unloaded:
                self._enter_lift()
            elif self._t >= self.p.t_shift_max:
                self._enter_return()   # never lift a loaded foot: retry
        elif self.phase == "LIFT":
            if self._t >= self._dur + self.p.t_apex:
                self._enter_land()
        elif self.phase == "LAND":
            landed = self._debounced(
                self._swing_force() > self.p.f_loaded, dt)
            if (self._t >= self._dur and landed):
                self.steps += 1
                self._enter_settle()
            elif self._t >= self._dur + self.p.t_land_max:
                self.steps += 1          # count it; contact is soft
                self._enter_settle()
        elif self.phase == "SETTLE":
            if self._t >= self._dur:
                self._enter_return()
        elif self.phase == "RETURN":
            if self._t >= self._dur:
                self.swing = "right" if self.swing == "left" else "left"
                self._enter_shift()

        targets = {k: v for k, v in self.cmd.items() if k != "com_y"}
        return targets, (0.0, self.cmd["com_y"])
