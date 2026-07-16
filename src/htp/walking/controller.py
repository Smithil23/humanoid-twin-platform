"""
controller.py — online LIPM/ZMP walking controller.

Ties together footstep planning, ZMP-preview CoM generation, swing-foot
trajectories, and leg IK into a per-tick control loop. Designed to drop
onto your existing stack:

  * sim.py     -> implement the RobotIO protocol (read state, send targets).
  * balance.py -> implement the Stabilizer protocol (feedforward + feedback).
                  Your capture-point / ankle / hip strategy plugs in here as
                  the feedback layer on top of this feedforward pattern.
  * stepper.py -> its event-based stepping can call `replan()` to inject or
                  shift footsteps for push recovery without stopping the walk.

The preview controller runs ONLINE (receding horizon): each tick it consumes
the ZMP reference for the current sample plus the preview window, so the
footstep plan can be edited mid-walk and the CoM pattern adapts within one
preview horizon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional

import numpy as np

try:
    from .preview import PreviewController
except ImportError:
    from preview import PreviewController
try:
    from .footsteps import (GaitParams, Side, Footstep, plan_footsteps,
                            support_sequence, zmp_reference)
except ImportError:
    from footsteps import (GaitParams, Side, Footstep, plan_footsteps,
                           support_sequence, zmp_reference)
try:
    from .swing import swing_foot_pose
except ImportError:
    from swing import swing_foot_pose
try:
    from .leg_ik import LegParams, leg_ik, Rz
except ImportError:
    from leg_ik import LegParams, leg_ik, Rz


# ---------------------------------------------------------------------------
# Adapter protocols — implement these against your sim.py / balance.py.
# ---------------------------------------------------------------------------
class RobotIO(Protocol):
    """Bridge to sim.py. Joint name order per leg must match leg_ik:
       [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]."""

    def read_com(self) -> np.ndarray: ...          # measured CoM (3,) world
    def read_zmp(self) -> np.ndarray: ...           # measured ZMP (2,) world
    def send_leg_targets(self, left_q: np.ndarray,
                         right_q: np.ndarray) -> None: ...


class Stabilizer(Protocol):
    """Bridge to balance.py. Returns a ZMP correction (2,) [m] to add to the
    reference — i.e. shift the desired ZMP toward the measured CoM error to
    reject disturbances (this is where your ankle/hip/capture-point logic
    lives). Return zeros for pure feedforward."""

    def zmp_correction(self, com_ref: np.ndarray, com_meas: np.ndarray,
                       zmp_ref: np.ndarray, zmp_meas: np.ndarray,
                       t: float) -> np.ndarray: ...


class _NullStabilizer:
    def zmp_correction(self, *_args) -> np.ndarray:
        return np.zeros(2)


# ---------------------------------------------------------------------------
@dataclass
class WalkConfig:
    gait: GaitParams = field(default_factory=GaitParams)
    leg: LegParams = field(default_factory=LegParams)
    n_steps: int = 12
    first_swing: Side = Side.LEFT
    preview_horizon: float = 1.6
    pelvis_z_above_com: float = 0.0   # if pelvis frame != CoM, offset here
    # Horizontal offset of the real CoM from the pelvis/hip reference point
    # (com - hip_mid), measured on the robot. The pelvis is placed at
    # (CoM_ref - this) so the ACTUAL CoM lands on the reference / over the
    # support feet. Without it, a crouch rocks the CoM off the feet.
    com_offset_xy: tuple = (0.0, 0.0)
    follow_heading: bool = True       # yaw pelvis along the path


class WalkController:
    def __init__(self, cfg: WalkConfig, io: RobotIO,
                 stabilizer: Optional[Stabilizer] = None,
                 start_pose: tuple = (0.0, 0.0, 0.0)) -> None:
        self.cfg = cfg
        self.io = io
        self.stab = stabilizer or _NullStabilizer()
        self.dt = cfg.gait.dt
        self.pc_x = PreviewController(self.dt, cfg.gait.com_height,
                                      cfg.preview_horizon)
        self.pc_y = PreviewController(self.dt, cfg.gait.com_height,
                                      cfg.preview_horizon)
        self.n_preview = self.pc_x.n_preview
        self._start_pose = start_pose
        self.replan(start_pose)
        self.k = 0

    # -- planning ----------------------------------------------------------
    def replan(self, start_pose: tuple, from_time: float = 0.0) -> None:
        """(Re)build footsteps and the ZMP reference. Call from stepper.py
        to adjust the plan mid-walk; the preview window absorbs the change."""
        self.steps = plan_footsteps(self.cfg.gait, self.cfg.n_steps,
                                    start_pose, self.cfg.first_swing)
        self.phases = support_sequence(self.steps, self.cfg.gait)
        t, zx, zy = zmp_reference(self.steps, self.cfg.gait)
        # pad the tail so the preview window never runs off the end
        pad = self.n_preview + 2
        self.zx = np.concatenate([zx, np.repeat(zx[-1], pad)])
        self.zy = np.concatenate([zy, np.repeat(zy[-1], pad)])
        self.N = len(zx)
        self.T_end = t[-1] if len(t) else 0.0
        # seed CoM at the initial ZMP so we start balanced
        self.pc_x.reset(zx[0])
        self.pc_y.reset(zy[0])

    def done(self) -> bool:
        return self.k >= self.N

    # -- phase lookup ------------------------------------------------------
    def _phase_at(self, t: float):
        for (t0, t1, kind, support, sf, st) in self.phases:
            if t0 <= t < t1:
                return t0, t1, kind, support, sf, st
        return self.phases[-1]

    def _foot_world_poses(self, t: float):
        """Return (left_pos, left_yaw, right_pos, right_yaw) for the feet."""
        t0, t1, kind, support, sf, st = self._phase_at(t)
        gait = self.cfg.gait

        # default: both feet at their most recent planted pose
        def planted(side: Side):
            best = None
            for s in self.steps:
                if s.side is side and s.t_touchdown <= t + 1e-9:
                    if best is None or s.t_touchdown >= best.t_touchdown:
                        best = s
            if best is None:  # fallback to the initial standing foot
                best = next(s for s in self.steps if s.side is side)
            return np.array([best.x, best.y, 0.0]), best.theta

        lpos, lyaw = planted(Side.LEFT)
        rpos, ryaw = planted(Side.RIGHT)

        if kind == "SS" and st is not None:
            # swinging foot follows the arc from its liftoff pose to `st`
            phase = (t - t0) / max(t1 - t0, 1e-6)
            prev_pos, prev_yaw = planted(st.side)
            p_from = prev_pos[:2]
            p_to = np.array([st.x, st.y])
            pos, yaw, _ = swing_foot_pose(phase, p_from, p_to,
                                          prev_yaw, st.theta,
                                          gait.step_height)
            if st.side is Side.LEFT:
                lpos, lyaw = pos, yaw
            else:
                rpos, ryaw = pos, yaw
        return lpos, lyaw, rpos, ryaw

    # -- main tick ---------------------------------------------------------
    def step(self) -> dict:
        """Advance one control tick. Returns a debug dict; also sends targets."""
        k, dt = self.k, self.dt
        t = k * dt

        # ZMP reference: current sample + preview window, with feedback shift.
        zmp_ref_now = np.array([self.zx[k], self.zy[k]])
        com_meas = self.io.read_com()
        zmp_meas = self.io.read_zmp()
        com_ref_prev = np.array([self.pc_x.x[0, 0], self.pc_y.x[0, 0]])
        corr = self.stab.zmp_correction(com_ref_prev, com_meas,
                                        zmp_ref_now, zmp_meas, t)

        win_x = self.zx[k + 1: k + 1 + self.n_preview] + corr[0]
        win_y = self.zy[k + 1: k + 1 + self.n_preview] + corr[1]
        cx, _, _, _ = self.pc_x.step(win_x, self.zx[k] + corr[0])
        cy, _, _, _ = self.pc_y.step(win_y, self.zy[k] + corr[1])
        cz = self.cfg.gait.com_height + self.cfg.pelvis_z_above_com

        # pelvis pose: shift by -com_offset so the ACTUAL CoM (= pelvis +
        # com_offset) lands on the reference (cx, cy), keeping it over the feet.
        ox, oy = self.cfg.com_offset_xy
        pelvis_pos = np.array([cx - ox, cy - oy, cz])
        yaw = 0.0
        if self.cfg.follow_heading:
            _, _, _, support, _, _ = self._phase_at(t)
            yaw = support.theta
        pelvis_rot = Rz(yaw)

        # hip bases: pelvis offset laterally by +/- hip_offset_y
        lp = self.cfg.leg
        left_hip = pelvis_pos + pelvis_rot @ np.array([0, +lp.hip_offset_y, 0])
        right_hip = pelvis_pos + pelvis_rot @ np.array([0, -lp.hip_offset_y, 0])

        # foot targets
        lpos, lyaw, rpos, ryaw = self._foot_world_poses(t)
        lrot = Rz(lyaw)
        rrot = Rz(ryaw)

        left_q = leg_ik(lpos, lrot, left_hip, pelvis_rot, lp)
        right_q = leg_ik(rpos, rrot, right_hip, pelvis_rot, lp)
        self.io.send_leg_targets(left_q, right_q)

        self.k += 1
        return {"t": t, "com": pelvis_pos, "zmp_ref": zmp_ref_now,
                "left_foot": lpos, "right_foot": rpos,
                "left_q": left_q, "right_q": right_q, "corr": corr}


# ---------------------------------------------------------------------------
# Runnable smoke test with a mock RobotIO (no physics — just exercises the
# full per-tick pipeline and checks joint targets stay finite & continuous).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    class MockIO:
        def __init__(self):
            self.last = None
        def read_com(self):
            return np.array([0.0, 0.0, 0.80])
        def read_zmp(self):
            return np.array([0.0, 0.0])
        def send_leg_targets(self, lq, rq):
            self.last = (lq, rq)

    cfg = WalkConfig(n_steps=10)
    cfg.gait.step_length = 0.18
    io = MockIO()
    ctrl = WalkController(cfg, io)

    logs = []
    prev_lq = prev_rq = None
    max_jump = 0.0
    while not ctrl.done():
        d = ctrl.step()
        logs.append(d)
        lq, rq = d["left_q"], d["right_q"]
        assert np.all(np.isfinite(lq)) and np.all(np.isfinite(rq)), "NaN joint!"
        if prev_lq is not None:
            max_jump = max(max_jump,
                           np.max(np.abs(lq - prev_lq)),
                           np.max(np.abs(rq - prev_rq)))
        prev_lq, prev_rq = lq, rq

    T = logs[-1]["t"]
    coms = np.array([d["com"] for d in logs])
    lf = np.array([d["left_foot"] for d in logs])
    rf = np.array([d["right_foot"] for d in logs])
    print(f"ran {len(logs)} ticks over {T:.2f}s at dt={cfg.gait.dt}")
    print(f"CoM x advanced: {coms[0,0]:+.3f} -> {coms[-1,0]:+.3f} m")
    print(f"CoM y sway range: [{coms[:,1].min():+.3f}, {coms[:,1].max():+.3f}] m")
    print(f"left foot peak lift:  {lf[:,2].max()*1000:.1f} mm")
    print(f"right foot peak lift: {rf[:,2].max()*1000:.1f} mm")
    print(f"max per-tick joint step: {np.degrees(max_jump):.3f} deg "
          f"(<~1 deg => smooth)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.array([d["t"] for d in logs])
        fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axs[0].plot(t, coms[:, 0], label="CoM x")
        axs[0].plot(t, coms[:, 1], label="CoM y")
        axs[0].plot(t, lf[:, 2], label="L foot z")
        axs[0].plot(t, rf[:, 2], label="R foot z")
        axs[0].legend(ncol=4); axs[0].grid(alpha=0.3)
        axs[0].set_ylabel("m"); axs[0].set_title("walk pattern")
        lq = np.array([d["left_q"] for d in logs])
        names = ["hipY", "hipR", "hipP", "knee", "ankP", "ankR"]
        for j in range(6):
            axs[1].plot(t, np.degrees(lq[:, j]), lw=1, label=names[j])
        axs[1].legend(ncol=6, fontsize=8); axs[1].grid(alpha=0.3)
        axs[1].set_ylabel("deg"); axs[1].set_xlabel("time [s]")
        axs[1].set_title("left leg joint targets")
        fig.tight_layout(); fig.savefig("controller_selftest.png", dpi=110)
        print("wrote controller_selftest.png")
    except Exception as exc:
        print("plot skipped:", exc)
