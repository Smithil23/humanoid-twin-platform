#!/usr/bin/env python3
"""
scripts/walk_lipm.py — LIPM/ZMP walking on STAR1.

Feedforward walking PATTERN (footstep plan + ZMP-preview CoM + swing-foot
arc + leg IK) from the robot-agnostic `htp.walking` package, stabilised by
your EXISTING, proven balance stack (htp.balance.BalanceController) applied
as an ankle/waist/arm offset overlay — the same overlay pattern march.py
uses, so the 162-step balance recipe carries straight over.

    LIPM controller  -> per-leg joint targets (feedforward foot placement)
    BalanceController-> ankle/waist offsets (capture-point feedback)  ── on top
    arm counter-swing-> shoulders (as in march.py)

Run these in order:
    python scripts/walk_lipm.py --list-joints   # (optional) verify joint names
    python scripts/walk_lipm.py --calibrate      # measure IK sign conventions
    #   -> paste the printed JOINT_SIGN / JOINT_OFFSET below
    python scripts/walk_lipm.py                   # walk (viewer, balance overlay on)
    python scripts/walk_lipm.py --no-balance      # feedforward only (debug)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# --- make `htp` importable when run as a loose script ---------------------
#   file: ...\htp\scripts\walk_lipm.py   package: ...\htp\src\htp
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402  (top-level: avoids shadowing `mujoco` in cmd_walk)

from htp import PlatformConfig, Simulator                # noqa: E402
from htp.balance import BalanceController                # noqa: E402
from htp.walking import WalkController, WalkConfig        # noqa: E402
from htp.walking.footsteps import Side                   # noqa: E402
from htp.walking.leg_ik import LegParams                 # noqa: E402
from htp.mj_ik import LegIK                               # noqa: E402  MuJoCo DLS IK

CONFIG_YAML = "configs/star1_visual.yaml"     # same as march.py
ARM_GAIN = 2.5                                  # shoulder counter-swing (march.py)
# Fraction of full leg length for the walk crouch. STAR1's stable stand is a
# near-straight leg, so start almost there (just off the singularity) and
# deepen later once walking is stable. 1.0 = full extension (stand).
CROUCH_FRAC = 0.995

# ==========================================================================
# Joint names, in leg_ik order: [hip_yaw, hip_roll, hip_pitch, knee,
# ankle_pitch, ankle_roll]. Pre-filled from your balance.py / stepper.py
# naming; verify with --list-joints if a name errors.
# ==========================================================================
LEG_JOINTS = {
    "left":  ["left_hip_yaw_joint",  "left_hip_roll_joint",  "left_hip_pitch_joint",
              "left_knee_joint",  "left_ankle_pitch_joint",  "left_ankle_roll_joint"],
    "right": ["right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
              "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
}
# Set by --calibrate. JOINT_SIGN holds the fitted linear SLOPE (usually +/-1)
# and JOINT_OFFSET the intercept: cmd = slope*ik + offset.
# --- calibrated for STAR1 (3-pose fit, max residual 0.11 deg) ---
JOINT_SIGN = {"left":  [1.0, 1.0, -1.0312, -1.0191, -1.0378, 1.0],
              "right": [1.0, 1.0, -1.0310, -1.0190, -1.0377, 1.0]}
JOINT_OFFSET = {"left":  [0.0, 0.0007, 0.5252, -1.0423, 0.5307, 0.0],
                "right": [0.0, 0.0007, 0.5252, -1.0423, 0.5307, 0.0]}


def build_config() -> PlatformConfig:
    return PlatformConfig.load(CONFIG_YAML)


# ------------------------------------------------------------------ helpers
def _jid(sim, name):
    return mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _bid(sim, name):
    return mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name)


def _yaw_from_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _side_of(link: str) -> str:
    return "left" if "left" in link else "right"


def measure_geometry(sim: Simulator) -> dict:
    """Auto-measure leg dimensions + stand heights from the live model.

    Segments are measured between the rigid-link endpoints hip_pitch -> knee
    -> ankle_pitch, so they equal the true thigh/shank at ANY pose (those
    anchors are the physical ends of each rigid link). The IK's hip point is
    the hip_pitch axis (STAR1's hip_yaw/roll sit ~0.14 m above it and only
    do yaw/roll, ~0 in straight walking).
    """
    mujoco.mj_forward(sim.model, sim.data)
    xa = sim.data.xanchor
    L, R = LEG_JOINTS["left"], LEG_JOINTS["right"]
    l_hp = xa[_jid(sim, L[2])]          # hip_pitch anchor = IK "hip" point
    r_hp = xa[_jid(sim, R[2])]
    knee = xa[_jid(sim, L[3])]
    ankle = xa[_jid(sim, L[4])]         # ankle_pitch anchor
    thigh = float(np.linalg.norm(l_hp - knee))
    shank = float(np.linalg.norm(knee - ankle))
    hip_off = float(abs(l_hp[1] - r_hp[1]) / 2.0)
    feet = list(sim.cfg.feet.links)
    fpos = {ln: sim.data.xpos[_bid(sim, ln)] for ln in feet}
    foot_sep = float(abs(list(fpos.values())[0][1]
                         - list(fpos.values())[1][1]) / 2.0)
    foot_z = float(np.mean([p[2] for p in fpos.values()]))
    com_z = float(sim.data.subtree_com[1][2])
    hip_pitch_z = float(l_hp[2])
    com_xy = sim.data.subtree_com[1][:2].copy()
    hip_mid_xy = np.array([(l_hp[0] + r_hp[0]) / 2.0, (l_hp[1] + r_hp[1]) / 2.0])
    return dict(thigh=thigh, shank=shank, hip_off=hip_off, foot_sep=foot_sep,
                foot_z=foot_z, com_z=com_z, hip_pitch_z=hip_pitch_z,
                com_xy=com_xy, hip_mid_xy=hip_mid_xy)


def make_leg_params(geo: dict) -> LegParams:
    return LegParams(thigh=geo["thigh"], shank=geo["shank"],
                     hip_offset_y=geo["hip_off"])


# --------------------------------------------------------------------- I/O
class Star1IO:
    """RobotIO for the walking package. Buffers leg targets instead of
    commanding directly, so the loop can overlay balance offsets before
    committing (see cmd_walk)."""

    def __init__(self, sim: Simulator):
        self.sim = sim
        self._bid = {ln: _bid(sim, ln) for ln in sim.cfg.feet.links}
        self.pending: dict[str, float] = {}

    def read_com(self) -> np.ndarray:
        return self.sim.data.subtree_com[1].copy()

    def read_zmp(self) -> np.ndarray:
        forces = self.sim.foot_forces()
        total = sum(forces.values())
        com_xy = self.sim.data.subtree_com[1][:2]
        if total < 1.0:
            return com_xy.copy()
        cop = np.zeros(2)
        for link, f in forces.items():
            cop += f * self.sim.data.xpos[self._bid[link]][:2]
        return cop / total

    def send_leg_targets(self, left_q, right_q) -> None:
        t = {}
        for side, q6 in (("left", left_q), ("right", right_q)):
            for name, val, sgn, off in zip(LEG_JOINTS[side], q6,
                                           JOINT_SIGN[side], JOINT_OFFSET[side]):
                t[name] = sgn * float(val) + off
        self.pending = t


class NullStabilizer:
    def zmp_correction(self, *_a):
        return np.zeros(2)


# ----------------------------------------------------------------- commands
def cmd_list_joints(sim: Simulator) -> None:
    print(f"\n{len(sim.hinge_names)} hinge joints:\n")
    for n in sim.hinge_names:
        print(f"    {n}")
    geo = measure_geometry(sim)
    print(f"\nmeasured: thigh={geo['thigh']:.4f} shank={geo['shank']:.4f} "
          f"hip_off={geo['hip_off']:.4f} foot_sep={geo['foot_sep']:.4f}")
    print(f"          stand com_z={geo['com_z']:.4f} hip_z={geo['hip_z']:.4f}")


def cmd_calibrate(sim: Simulator) -> None:
    """Two-pose sign/offset calibration.

    STAR1's stand pose is a geometrically STRAIGHT leg (my IK -> ~0 there), so
    a single pose can't resolve joint signs. We sample two poses -- the settled
    stand and a deliberately deeper crouch -- and get each joint's sign from
    the DIRECTION its actual angle moves vs the direction my IK moves:

        actual = sign * my_ik + offset
        sign   = sign( d(actual) / d(my_ik) )     offset = actual_A - sign*my_A

    A self-check residual (predicted actual_B vs measured) confirms validity."""
    from htp.walking.leg_ik import leg_ik
    names = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_p", "ankle_r"]

    def actual_q(name):
        j = _jid(sim, name)
        return float(sim.data.qpos[sim.model.jnt_qposadr[j]])

    def snapshot():
        """Run IK on the current kinematic pose; return per-side (my_q, act_q)."""
        mujoco.mj_forward(sim.model, sim.data)
        geo = measure_geometry(sim)
        lp = make_leg_params(geo)
        pelvis_rot = sim.data.xmat[_bid(sim, sim.cfg.robot.root_link)].reshape(3, 3)
        out = {}
        for side in ("left", "right"):
            hip = sim.data.xanchor[_jid(sim, LEG_JOINTS[side][2])]   # hip_PITCH
            fl = [ln for ln in sim.cfg.feet.links if _side_of(ln) == side][0]
            fb = _bid(sim, fl)
            my_q = leg_ik(sim.data.xpos[fb], sim.data.xmat[fb].reshape(3, 3),
                          hip, pelvis_rot, lp)
            act = np.array([actual_q(n) for n in LEG_JOINTS[side]])
            out[side] = (np.asarray(my_q), act)
        return geo, out

    # --- Pose A: settle into the real stand ------------------------------
    sim.reset()
    dt = float(sim.cfg.sim.timestep)
    for _ in range(int(2.0 / dt)):
        mujoco.mj_step(sim.model, sim.data)
    geoA, poseA = snapshot()

    # diagnostic geometry dump
    print("\n=== raw settled geometry (left leg, world XYZ) ===")
    for lbl, jn in zip(names, LEG_JOINTS["left"]):
        p = sim.data.xanchor[_jid(sim, jn)]
        print(f"     {lbl:10s} anchor = [{p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}]  ({jn})")
    print(f"     measured: thigh={geoA['thigh']:.4f} shank={geoA['shank']:.4f} "
          f"(sum={geoA['thigh']+geoA['shank']:.4f})  hip_off={geoA['hip_off']:.4f}")
    print(f"     settled: com_z={geoA['com_z']:.4f} "
          f"hip_pitch_z={geoA['hip_pitch_z']:.4f} foot_z={geoA['foot_z']:.4f}")

    # --- Poses B,C: progressively deeper crouches ------------------------
    # (scale the pitch chain from stand; each gives another (my, actual)
    #  sample so a per-joint line can be fit and its residual checked.)
    saved = sim.data.qpos.copy()
    PITCH_IDX = [2, 3, 4]                 # hip_pitch, knee, ankle_pitch
    poses = {side: [poseA[side]] for side in ("left", "right")}
    for scale in (1.15, 1.30):
        sim.data.qpos[:] = saved
        for side in ("left", "right"):
            for i in PITCH_IDX:
                adr = sim.model.jnt_qposadr[_jid(sim, LEG_JOINTS[side][i])]
                sim.data.qpos[adr] = float(saved[adr]) * scale
        _, pose = snapshot()
        for side in ("left", "right"):
            poses[side].append(pose[side])
    sim.data.qpos[:] = saved
    mujoco.mj_forward(sim.model, sim.data)

    # --- fit actual = slope*my + offset per joint (least squares) --------
    print("\n=== IK calibration (3-pose linear fit) ===")
    sign_out, off_out = {}, {}
    for side in ("left", "right"):
        mys = np.array([s[0] for s in poses[side]])     # (3, 6)
        acts = np.array([s[1] for s in poses[side]])    # (3, 6)
        slopes, offs, resids = [], [], []
        print(f"\n[{side}] joint      slope   offset   resid(deg)  samples my->act")
        for k, n in enumerate(names):
            x, y = mys[:, k], acts[:, k]
            if float(np.var(x)) < 1e-8:                 # stationary joint
                sl, of = 1.0, float(np.mean(y) - np.mean(x))
            else:
                sl, of = np.polyfit(x, y, 1)
            pred = sl * x + of
            rsd = float(np.max(np.abs(pred - y)))
            slopes.append(round(float(sl), 4))
            offs.append(round(float(of), 4))
            resids.append(rsd)
            smp = " ".join(f"{xi:+.2f}->{yi:+.2f}" for xi, yi in zip(x, y))
            print(f"    {n:9s} {sl:+6.3f} {of:+7.3f}  {np.degrees(rsd):7.2f}   {smp}")
        worst = np.degrees(max(resids))
        tag = "OK" if worst < 3.0 else "HIGH - fit/correspondence suspect"
        print(f"    max fit residual: {worst:.2f} deg  [{tag}]")
        sign_out[side] = slopes
        off_out[side] = offs
    print("\nPaste into walk_lipm.py (JOINT_SIGN holds the fitted slope):")
    print(f"JOINT_SIGN = {{\"left\": {sign_out['left']}, "
          f"\"right\": {sign_out['right']}}}")
    print(f"JOINT_OFFSET = {{\"left\": {off_out['left']}, "
          f"\"right\": {off_out['right']}}}")
    print("\n(Then: --stand to verify, before a full walk.)")


def _build_walk(sim, geo, n_steps, step_len):
    lp = make_leg_params(geo)                       # thigh/shank ~0.36 each
    cfg = WalkConfig(n_steps=n_steps, leg=lp, first_swing=Side.LEFT)
    cfg.gait.dt = float(sim.model.opt.timestep)     # match physics rate (march.py)
    cfg.gait.step_length = step_len
    cfg.gait.step_width = geo["foot_sep"]           # start at actual stance width
    leg_len = geo["thigh"] + geo["shank"]
    # z_c = real CoM height above the foot; hip placed at a CROUCHED height so
    # the knees stay bent (reach margin, no straight-leg singularity). The
    # crouch is the real stand hip height, capped at 90% of full extension.
    com_above_foot = geo["com_z"] - geo["foot_z"]
    # Crouch only slightly below full extension: STAR1's stable stand is a
    # near-straight leg (my_knee ~0), and calibration only covered a shallow
    # bend, so a deep crouch both leaves the calibrated range and is unstable.
    # 0.985 keeps the knee ~20 deg bent (out of singularity) yet near stand.
    hip_above_foot = min(geo["hip_pitch_z"] - geo["foot_z"], CROUCH_FRAC * leg_len)
    cfg.gait.com_height = com_above_foot                       # z_c for LIPM
    cfg.pelvis_z_above_com = hip_above_foot - com_above_foot   # -> IK hip crouch
    return cfg


def _settle(sim, dt, secs=2.0):
    sim.reset()
    for _ in range(int(secs / dt)):
        mujoco.mj_step(sim.model, sim.data)
    mujoco.mj_forward(sim.model, sim.data)


def _feet_center_xy(sim):
    return np.mean([sim.data.xpos[_bid(sim, ln)][:2]
                    for ln in sim.cfg.feet.links], axis=0)


def _solve_crouch_offset(sim, base, dt, n_steps, step_len, iters=3):
    """Find the pelvis-shift (com_offset_xy) that keeps the ACTUAL CoM over
    the feet in the crouch. The analytic IK + crouch lean puts the CoM behind
    the feet; we blend into the crouch, measure com-minus-feet, add it to the
    offset, and repeat until the CoM sits over the feet."""
    off = np.zeros(2)
    for it in range(iters):
        _settle(sim, dt)
        geo = measure_geometry(sim)
        cfg = _build_walk(sim, geo, n_steps, step_len)
        cfg.com_offset_xy = (float(off[0]), float(off[1]))
        io = Star1IO(sim)
        sxy = sim.data.subtree_com[1][:2].copy()
        yaw0 = _yaw_from_quat(sim.data.qpos[3:7])
        ctrl = WalkController(cfg, io, NullStabilizer(),
                              start_pose=(float(sxy[0]), float(sxy[1]), yaw0))
        ctrl.step()
        _blend_to(sim, base, dict(io.pending), dt, secs=1.0)
        resid = sim.data.subtree_com[1][:2] - _feet_center_xy(sim)
        off = off + resid
        print(f"  crouch-offset iter {it}: residual="
              f"({resid[0]:+.3f},{resid[1]:+.3f}) -> offset="
              f"({off[0]:+.3f},{off[1]:+.3f})")
    return off


def _blend_to(sim, base, leg_target, dt, secs=1.2):
    """Ramp leg joints from their current angles to leg_target (cosine),
    so the robot eases into the walk crouch instead of snapping (which
    lifts the feet and drops the robot)."""
    cur = {n: float(sim.data.qpos[sim.model.jnt_qposadr[_jid(sim, n)]])
           for n in leg_target}
    n_steps = max(1, int(secs / dt))
    for i in range(n_steps):
        a = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / n_steps))
        tgt = dict(base)
        for n, v in leg_target.items():
            tgt[n] = cur[n] + a * (v - cur[n])
        sim.set_joint_targets(tgt)
        sim.step()


def build_ik_solver(sim):
    """Build a MuJoCo DLS leg-IK solver from the live model + config."""
    foot_bodies = {s: [ln for ln in sim.cfg.feet.links if _side_of(ln) == s][0]
                   for s in ("left", "right")}
    leg_joints = {s: LEG_JOINTS[s] for s in ("left", "right")}
    f = sim.cfg.feet
    # sole contact point in the foot-body frame: box centre offset, minus
    # half the box thickness (the bottom face that touches the floor).
    sole_off = np.array([f.offset[0], f.offset[1], f.offset[2] - f.size[2] / 2.0])
    return LegIK(sim.model, foot_bodies, leg_joints, sole_off), leg_joints, sole_off


def _measured_sole(sim, side, sole_off):
    ln = [l for l in sim.cfg.feet.links if _side_of(l) == side][0]
    bid = _bid(sim, ln)
    R = sim.data.xmat[bid].reshape(3, 3).copy()
    p = sim.data.xpos[bid].copy() + R @ sole_off
    return p, R


def _flat_R(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def cmd_test_ik(sim: Simulator) -> None:
    """Validate the MuJoCo IK on the REAL STAR1 model (document #2 test):
    generate reachable sole targets, solve, and report Cartesian/angular
    error. Target: < 2 mm and < 0.5 deg."""
    _settle(sim, float(sim.cfg.sim.timestep))
    ik, leg_joints, sole_off = build_ik_solver(sim)
    q0 = sim.data.qpos.copy()
    pel_z0 = float(q0[2])
    rng = np.random.default_rng(0)
    Lp, _ = _measured_sole(sim, "left", sole_off)
    Rp, _ = _measured_sole(sim, "right", sole_off)
    print(f"\nmeasured soles: L={Lp.round(3)} R={Rp.round(3)}  pelvis_z={pel_z0:.3f}")
    maxp = maxr = 0.0
    nok = 0
    N = 200
    for _ in range(N):
        pz = pel_z0 - rng.uniform(0.02, 0.10)          # crouch range
        tgts = {}
        for s, base_p in (("left", Lp), ("right", Rp)):
            dx, dy = rng.uniform(-0.08, 0.08), rng.uniform(-0.03, 0.03)
            dz = rng.uniform(0.0, 0.05)
            yaw = rng.uniform(-0.1, 0.1)
            tgts[s] = (base_p + np.array([dx, dy, dz]), _flat_R(yaw))
        _, info = ik.solve(q0, [q0[0], q0[1], pz], q0[3:7], tgts, iters=80)
        ok = all(info[s]["pos_mm"] < 2 and info[s]["rot_deg"] < 0.5
                 for s in ("left", "right"))
        for s in ("left", "right"):
            maxp = max(maxp, info[s]["pos_mm"])
            maxr = max(maxr, info[s]["rot_deg"])
        nok += ok
    print(f"IK validation ({N} reachable targets): converged {nok}/{N}")
    print(f"  max position error: {maxp:.3f} mm   (target < 2 mm)")
    print(f"  max orientation error: {maxr:.4f} deg (target < 0.5 deg)")
    print("RESULT:", "PASS" if (maxp < 2 and maxr < 0.5 and nok == N) else "CHECK")


def cmd_stand(sim: Simulator, hold_s: float = 5.0, crouch: float = 0.06,
              use_balance: bool = True) -> None:
    """Crouch-and-hold using MuJoCo IK, descending in TASK space so the feet
    stay pinned to the floor the whole time (joint-angle blending lifts them
    and topples the robot). CoM is kept over the feet at every height."""
    dt = float(sim.cfg.sim.timestep)
    _settle(sim, dt)
    ik, leg_joints, sole_off = build_ik_solver(sim)
    q0 = sim.data.qpos.copy()
    yaw0 = _yaw_from_quat(q0[3:7])
    feet_c = _feet_center_xy(sim)
    tgts = {s: (_measured_sole(sim, s, sole_off)[0], _flat_R(yaw0))
            for s in ("left", "right")}

    # Precompute the descent: pelvis from current height down by `crouch`,
    # IK solved with feet pinned + CoM over feet at each step.
    z0 = float(q0[2])
    n_kf = 40
    zs = z0 - crouch * 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n_kf)))
    keyframes = []
    qseed = q0.copy()
    for z in zs:
        angles, info = ik.solve_with_com(qseed, [feet_c[0], feet_c[1], z],
                                         q0[3:7], tgts, com_target_xy=feet_c,
                                         com_iters=8, iters=80)
        leg = {leg_joints[s][k]: float(angles[s][k])
               for s in ("left", "right") for k in range(6)}
        keyframes.append(leg)
    print(f"\nMuJoCo-IK stand: crouch {crouch*100:.0f}cm, {n_kf} keyframes, "
          f"final CoM err={info['com_err']*1e3:.2f}mm, "
          f"pelvis shifted x={info['pelvis_xy'][0]-feet_c[0]:+.3f}m")

    base = dict(sim.cfg.poses.stand)
    bal = BalanceController(sim, hip_kp=2.5, hip_kd=0.5, hip_max=0.35)
    bal.reset()
    foot_links = list(sim.cfg.feet.links)
    foot_bids = {ln: _bid(sim, ln) for ln in foot_links}

    def diag(tag):
        ff = sim.foot_forces()
        com = sim.data.subtree_com[1]
        parts = [f"{_side_of(ln)[0].upper()}:z={sim.data.xpos[foot_bids[ln]][2]:+.3f},"
                 f"F={ff[ln]:4.0f}" for ln in foot_links]
        print(f"  [{tag:>6}] base_z={sim.data.qpos[2]:+.3f} comx={com[0]:+.3f} "
              f"comy={com[1]:+.3f} ncon={sim.data.ncon}  {'  '.join(parts)}")

    def command(leg):
        tgt = dict(base); tgt.update(leg)
        if use_balance:
            op, orr = bal.update(dt, ref=(0.0, 0.0))
            hp, hr = bal.hip_offsets(); sides = bal.stance_sides()
            for j in BalanceController.ANKLE_PITCH:
                if j.split("_")[0] in sides:
                    tgt[j] = tgt.get(j, 0.0) + op
            for j in BalanceController.ANKLE_ROLL:
                if j.split("_")[0] in sides:
                    tgt[j] = tgt.get(j, 0.0) + orr
            tgt["waist_pitch_joint"] = tgt.get("waist_pitch_joint", 0.0) + hp
            tgt["waist_roll_joint"] = tgt.get("waist_roll_joint", 0.0) + hr
        sim.set_joint_targets(tgt); sim.step()

    try:
        vctx = mujoco.viewer.launch_passive(sim.model, sim.data)
    except Exception:
        vctx = None

    ticks_per_kf = max(1, int((1.6 / n_kf) / dt))    # ~1.6s total descent

    def run_loop(v=None):
        diag("start")
        for kf in keyframes:                          # descend, feet pinned
            for _ in range(ticks_per_kf):
                command(kf)
                if v is not None:
                    v.sync(); time.sleep(max(0, dt))
        diag("crouch")
        for i in range(int(hold_s / dt)):             # hold
            command(keyframes[-1])
            if v is not None:
                v.sync(); time.sleep(max(0, dt))
            if i % int(1.0 / dt) == 0:
                diag(f"{i*dt:.1f}s")

    if vctx is not None:
        with vctx as v:
            run_loop(v)
    else:
        run_loop()
    st = sim.state()
    print(f"after hold: base_height={st.base_height:.3f} upright={sim.upright} "
          f"margin={sim.balance_margin():+.3f}")


def cmd_walk(sim: Simulator, n_steps: int, step_len: float,
             use_balance: bool) -> None:
    sim.reset()
    dt = float(sim.cfg.sim.timestep)
    for _ in range(int(2.0 / dt)):                  # settle at stand (march.py)
        mujoco.mj_step(sim.model, sim.data)
    mujoco.mj_forward(sim.model, sim.data)

    geo = measure_geometry(sim)
    cfg = _build_walk(sim, geo, n_steps, step_len)

    # start the plan at the robot's ACTUAL stand pose so LIPM coords == sim
    # world coords (keeps the balance `ref` frame-consistent).
    start_xy = sim.data.subtree_com[1][:2].copy()  # CoM ground point
    yaw0 = _yaw_from_quat(sim.data.qpos[3:7])
    io = Star1IO(sim)
    ctrl = WalkController(cfg, io, NullStabilizer(),
                          start_pose=(float(start_xy[0]), float(start_xy[1]), yaw0))
    bal = BalanceController(sim, hip_kp=2.5, hip_kd=0.5, hip_max=0.35)
    bal.reset()
    base = dict(sim.cfg.poses.stand)

    print(f"walk: {n_steps} steps, step_len={step_len} m, dt={cfg.gait.dt}, "
          f"com_z={cfg.gait.com_height:.3f}, hip_pitch_z={geo['hip_pitch_z']:.3f}, "
          f"balance={'on' if use_balance else 'OFF'}")

    # Ease into the walk crouch first, then start the controller fresh from
    # the blended pose (avoids the snap that lifts the feet and drops the bot).
    ctrl.step()
    _blend_to(sim, base, dict(io.pending), dt, secs=1.2)
    start_xy = sim.data.subtree_com[1][:2].copy()  # CoM ground point
    yaw0 = _yaw_from_quat(sim.data.qpos[3:7])
    ctrl = WalkController(cfg, io, NullStabilizer(),
                          start_pose=(float(start_xy[0]), float(start_xy[1]), yaw0))
    bal.reset()

    try:
        viewer_ctx = mujoco.viewer.launch_passive(sim.model, sim.data)
    except Exception as exc:
        print(f"(no viewer: {exc}) running headless")
        viewer_ctx = None

    def loop_body() -> bool:
        d = ctrl.step()                              # buffers io.pending
        tgt = dict(base)
        tgt.update(io.pending)                       # feedforward leg targets
        if use_balance:
            desired = d["com"][:2]
            center = sim.support_polygon().mean(axis=0)
            ref = tuple(np.asarray(desired) - center)
            op, orr = bal.update(dt, ref=ref)
            hp, hr = bal.hip_offsets()
            sides = bal.stance_sides()
            for j in BalanceController.ANKLE_PITCH:
                if j.split("_")[0] in sides:
                    tgt[j] = tgt.get(j, 0.0) + op
            for j in BalanceController.ANKLE_ROLL:
                if j.split("_")[0] in sides:
                    tgt[j] = tgt.get(j, 0.0) + orr
            tgt["waist_pitch_joint"] = tgt.get("waist_pitch_joint", 0.0) + hp
            tgt["waist_roll_joint"] = tgt.get("waist_roll_joint", 0.0) + hr
            for sh in ("left_shoulder_roll_joint", "right_shoulder_roll_joint"):
                tgt[sh] = tgt.get(sh, 0.0) + ARM_GAIN * hr
        sim.set_joint_targets(tgt)
        sim.step()
        if not sim.upright and sim.state().base_height < 0.3:
            print(f"  FELL at t={sim.data.time:.2f}s")
            return False
        return True

    if viewer_ctx is not None:
        with viewer_ctx as v:
            while v.is_running() and not ctrl.done():
                t0 = time.time()
                if not loop_body():
                    break
                v.sync()
                slack = dt - (time.time() - t0)
                if slack > 0:
                    time.sleep(slack)
    else:
        while not ctrl.done():
            if not loop_body():
                break

    st = sim.state()
    print(f"done. base_height={st.base_height:.3f} upright={sim.upright} "
          f"com_x={st.com[0]:+.3f} margin={sim.balance_margin():+.3f}")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="LIPM/ZMP walk on STAR1")
    ap.add_argument("--list-joints", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--test-ik", action="store_true",
                    help="validate MuJoCo IK accuracy on the real model")
    ap.add_argument("--stand", action="store_true",
                    help="hold the IK stand pose with balance (no stepping)")
    ap.add_argument("--no-balance", action="store_true",
                    help="feedforward only (debug the pattern / signs)")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--step-len", type=float, default=0.14)
    ap.add_argument("--crouch", type=float, default=0.06,
                    help="stand crouch depth in metres (task-space descent)")
    args = ap.parse_args()

    sim = Simulator(build_config())
    if args.list_joints:
        cmd_list_joints(sim)
    elif args.calibrate:
        cmd_calibrate(sim)
    elif getattr(args, "test_ik", False):
        cmd_test_ik(sim)
    elif args.stand:
        cmd_stand(sim, crouch=args.crouch, use_balance=not args.no_balance)
    else:
        cmd_walk(sim, args.steps, args.step_len, use_balance=not args.no_balance)


if __name__ == "__main__":
    main()
