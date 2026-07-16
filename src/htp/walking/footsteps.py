"""
footsteps.py — footstep planner and ZMP reference generation.

Plans an alternating sequence of foot placements for straight, turning,
or curved walking, then builds the piecewise ZMP reference (support-foot
centre during single support, linear ramp across double support) that
preview.py turns into a CoM trajectory.

Frames: world XY on the ground plane, theta = yaw about world Z (CCW+).
Foot placement is the desired ankle/sole centre projected on the ground.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Side(Enum):
    LEFT = "L"
    RIGHT = "R"

    @property
    def other(self) -> "Side":
        return Side.RIGHT if self is Side.LEFT else Side.LEFT

    @property
    def sign(self) -> float:
        # +1 places the foot to the robot's left of the path centreline.
        return +1.0 if self is Side.LEFT else -1.0


@dataclass
class Footstep:
    x: float
    y: float
    theta: float
    side: Side
    t_touchdown: float          # time this foot becomes the support foot
    t_liftoff: float            # time this foot leaves the ground next


@dataclass
class GaitParams:
    step_length: float = 0.18       # forward advance per step [m]
    step_width: float = 0.11        # half-distance between feet [m]
    step_time: float = 0.60         # single-support duration [s]
    double_support_time: float = 0.15   # DS duration [s]
    step_height: float = 0.05       # swing foot peak clearance [m]
    turn_per_step: float = 0.0      # yaw change per step [rad]
    com_height: float = 0.72        # z_c for the LIPM [m] (knees bent)
    dt: float = 0.005               # control timestep [s]


def plan_footsteps(gait: GaitParams,
                   n_steps: int,
                   start_pose: tuple = (0.0, 0.0, 0.0),
                   first_swing: Side = Side.LEFT) -> list[Footstep]:
    """Plan an alternating footstep sequence.

    The robot starts in double support with both feet straddling
    start_pose. `first_swing` is the first foot to leave the ground, so
    the first *support* foot is its opposite.

    Returns a list of Footstep, time-ordered by touchdown, including the
    two initial standing feet.
    """
    x0, y0, th0 = start_pose
    cycle = gait.step_time + gait.double_support_time

    def place(cx, cy, cth, side: Side):
        # offset a foot laterally from the path centreline point
        ox = -np.sin(cth) * gait.step_width * side.sign
        oy = +np.cos(cth) * gait.step_width * side.sign
        return cx + ox, cy + oy, cth

    steps: list[Footstep] = []

    # Two initial standing feet (support first, then the first swing foot),
    # both already on the ground at t = 0.
    support0 = first_swing.other
    sx, sy, sth = place(x0, y0, th0, support0)
    steps.append(Footstep(sx, sy, sth, support0,
                          t_touchdown=0.0, t_liftoff=np.inf))
    fx, fy, fth = place(x0, y0, th0, first_swing)
    steps.append(Footstep(fx, fy, fth, first_swing,
                          t_touchdown=0.0, t_liftoff=gait.double_support_time))

    # Advancing steps.
    cx, cy, cth = x0, y0, th0
    swing = first_swing
    t = gait.double_support_time            # first liftoff already scheduled
    for i in range(n_steps):
        # advance the centreline point along current heading, then turn
        cth_new = cth + gait.turn_per_step
        adv = gait.step_length if i > 0 else gait.step_length * 0.5
        cx += np.cos(cth_new) * adv
        cy += np.sin(cth_new) * adv
        cth = cth_new
        px, py, pth = place(cx, cy, cth, swing)

        t_td = t + gait.step_time
        steps.append(Footstep(px, py, pth, swing,
                              t_touchdown=t_td, t_liftoff=np.inf))
        # this foot lifts again one full cycle later (unless it's the last)
        t = t_td + gait.double_support_time
        swing = swing.other

    # Fill each footstep's liftoff = the touchdown of its next same-side step.
    for i, fs in enumerate(steps):
        for j in range(i + 1, len(steps)):
            if steps[j].side is fs.side:
                fs.t_liftoff = steps[j].t_touchdown - gait.double_support_time
                break

    return steps


def support_sequence(steps: list[Footstep], gait: GaitParams):
    """Derive support phases from the footstep list.

    Yields tuples (t_start, t_end, phase, support, swing_from, swing_to)
    where phase is 'DS' (double) or 'SS' (single). support/swing_* are
    Footstep objects (swing_* None during DS bookends).
    """
    # Order advancing (non-initial) steps by touchdown.
    advancing = [s for s in steps if s.t_touchdown > 0.0]
    advancing.sort(key=lambda s: s.t_touchdown)
    initial = [s for s in steps if s.t_touchdown == 0.0]

    phases = []
    ds = gait.double_support_time
    ss = gait.step_time

    # initial DS: ZMP under both starting feet, shifting toward first support
    first = advancing[0]
    prev_support = [s for s in initial if s.side is first.side.other][0]
    phases.append((0.0, ds, "DS", prev_support, None, None))

    t = ds
    support = prev_support
    for nxt in advancing:
        # SS: robot stands on `support`, swings the other foot to `nxt`
        phases.append((t, t + ss, "SS", support, support, nxt))
        t += ss
        # DS: weight transfers from `support` to the just-landed `nxt`
        phases.append((t, t + ds, "DS", nxt, None, None))
        t += ds
        support = nxt
    return phases


def zmp_reference(steps: list[Footstep], gait: GaitParams):
    """Build the ZMP reference at control rate from the support sequence.

    Single support: ZMP held at the support foot centre.
    Double support: ZMP ramps linearly from the outgoing to the incoming
                    support foot centre (keeps CoM jerk bounded).

    Returns (t, zmp_x, zmp_y) numpy arrays.
    """
    phases = support_sequence(steps, gait)
    T_end = phases[-1][1]
    N = int(round(T_end / gait.dt))
    t = np.arange(N) * gait.dt
    zx = np.zeros(N)
    zy = np.zeros(N)

    # Precompute per-phase endpoints.
    def foot_xy(fs: Footstep):
        return np.array([fs.x, fs.y])

    # Walk through phases and fill.
    # For DS we need the outgoing support (previous phase's support). Seed it
    # with the CENTRE of the two initial standing feet so the very first DS
    # ramps from a centred CoM to the first support foot -- otherwise the CoM
    # would start planted over one foot, splaying the legs and toppling the
    # robot at t=0 (there is no prior step to have shifted it there).
    initial_feet = [s for s in steps if s.t_touchdown == 0.0]
    if initial_feet:
        prev_support_xy = np.mean([foot_xy(s) for s in initial_feet], axis=0)
    else:
        prev_support_xy = None
    for idx, (t0, t1, kind, support, _sf, _st) in enumerate(phases):
        i0 = int(round(t0 / gait.dt))
        i1 = min(int(round(t1 / gait.dt)), N)
        if kind == "SS":
            p = foot_xy(support)
            zx[i0:i1] = p[0]
            zy[i0:i1] = p[1]
            prev_support_xy = p
        else:  # DS ramp from prev support to this phase's (incoming) support
            p_to = foot_xy(support)
            p_from = prev_support_xy if prev_support_xy is not None else p_to
            n = max(i1 - i0, 1)
            s = np.linspace(0.0, 1.0, n)
            zx[i0:i1] = p_from[0] + s * (p_to[0] - p_from[0])
            zy[i0:i1] = p_from[1] + s * (p_to[1] - p_from[1])
            prev_support_xy = p_to
    return t, zx, zy


# ----------------------------------------------------------------------
if __name__ == "__main__":
    from preview import PreviewController, generate_com_trajectory

    gait = GaitParams(step_length=0.18, step_width=0.11, step_time=0.60,
                      double_support_time=0.15, turn_per_step=0.0)
    steps = plan_footsteps(gait, n_steps=8, first_swing=Side.LEFT)
    print(f"planned {len(steps)} footsteps:")
    for s in steps[:6]:
        print(f"  {s.side.value}  x={s.x:+.3f} y={s.y:+.3f} "
              f"th={s.theta:+.2f}  td={s.t_touchdown:.2f} lo={s.t_liftoff:.2f}")

    t, zx, zy = zmp_reference(steps, gait)
    pc = PreviewController(dt=gait.dt, z_c=gait.com_height, preview_horizon=1.6)
    out_x = generate_com_trajectory(pc, zx, c0=zx[0])
    out_y = generate_com_trajectory(pc, zy, c0=zy[0])

    warm = pc.n_preview
    ex = out_x["zmp"][warm:-warm] - zx[warm:-warm]
    ey = out_y["zmp"][warm:-warm] - zy[warm:-warm]
    print(f"\nsteady-state ZMP tracking (excl. warm-up):")
    print(f"  x RMS {np.sqrt(np.mean(ex**2))*1000:.2f} mm  "
          f"max {np.max(np.abs(ex))*1000:.2f} mm")
    print(f"  y RMS {np.sqrt(np.mean(ey**2))*1000:.2f} mm  "
          f"max {np.max(np.abs(ey))*1000:.2f} mm")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        # top view
        axs[0].plot(out_x["com"], out_y["com"], "C0", lw=1.5, label="CoM")
        axs[0].plot(zx, zy, "k--", lw=0.8, label="ZMP ref")
        for s in steps:
            c = "C3" if s.side is Side.RIGHT else "C2"
            axs[0].plot(s.x, s.y, "s", color=c, ms=9)
        axs[0].set_aspect("equal"); axs[0].grid(alpha=0.3)
        axs[0].set_xlabel("x [m]"); axs[0].set_ylabel("y [m]")
        axs[0].set_title("top view: footsteps, ZMP ref, CoM path")
        axs[0].legend()
        # lateral vs time
        axs[1].plot(t, zy, "k--", lw=1, label="ZMP_y ref")
        axs[1].plot(t, out_y["zmp"], "C3", lw=1, label="ZMP_y")
        axs[1].plot(t, out_y["com"], "C0", lw=1.5, label="CoM_y")
        axs[1].set_xlabel("time [s]"); axs[1].set_ylabel("y [m]")
        axs[1].grid(alpha=0.3); axs[1].legend()
        axs[1].set_title("lateral ZMP tracking")
        fig.tight_layout(); fig.savefig("footsteps_selftest.png", dpi=110)
        print("wrote footsteps_selftest.png")
    except Exception as exc:
        print("plot skipped:", exc)
