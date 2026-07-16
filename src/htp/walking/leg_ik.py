"""
leg_ik.py — analytic 6-DOF leg inverse kinematics.

Standard humanoid leg topology (6 joints, hip-to-foot):
    q0 hip yaw   (Z)
    q1 hip roll  (X)
    q2 hip pitch (Y)
    q3 knee pitch(Y)     -- with thigh length A above, shank length B below
    q4 ankle pitch(Y)
    q5 ankle roll (X)

Given the hip-base pose (from the pelvis) and the desired foot pose, both
in world frame, returns the six joint angles. Correctness is verified by
FK round-trip in __main__ — if you retune LegParams to STAR1's URDF, rerun
this file and confirm the round-trip still passes.

IMPORTANT: this is a *reference* solver for the common leg layout. If your
balance stack already exposes leg IK matched to STAR1 (it likely does,
since footstep-based balance needs it), prefer that and treat this as a
fallback / cross-check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class LegParams:
    thigh: float = 0.40        # A: hip-pitch axis to knee axis [m]
    shank: float = 0.40        # B: knee axis to ankle axis [m]
    hip_offset_y: float = 0.11 # pelvis origin to hip, lateral [m] (+left)
    knee_forward: float = +1.0 # +1 = knee bends forward (human), else -1


def leg_fk(q: np.ndarray, hip_pos: np.ndarray, hip_rot: np.ndarray,
           lp: LegParams) -> tuple:
    """Forward kinematics: joint angles -> foot (pos, rot) in world."""
    q0, q1, q2, q3, q4, q5 = q
    R = hip_rot @ Rz(q0) @ Rx(q1) @ Ry(q2)
    p = hip_pos.copy()
    p = p + R @ np.array([0, 0, -lp.thigh])      # down the thigh
    R = R @ Ry(q3)
    p = p + R @ np.array([0, 0, -lp.shank])      # down the shank
    R = R @ Ry(q4) @ Rx(q5)                       # ankle
    return p, R


def leg_ik(foot_pos: np.ndarray, foot_rot: np.ndarray,
           hip_pos: np.ndarray, hip_rot: np.ndarray,
           lp: LegParams) -> np.ndarray:
    """Analytic IK. Returns q (6,). Clamps at the reachable limits (a nearly
    straight or fully folded leg is a singularity, not an error); raises only
    on gross (>3%) violations that signal a real geometry/target bug."""
    A, B = lp.thigh, lp.shank

    # Vector from foot (ankle) to hip, in foot frame.
    r = foot_rot.T @ (hip_pos - foot_pos)
    C = float(np.linalg.norm(r))
    reach_max, reach_min = A + B, abs(A - B)
    if C > reach_max * 1.03 or C < reach_min * 0.5:
        raise ValueError(
            f"leg target far out of range: reach {C:.3f} "
            f"(limits {reach_min:.3f}..{reach_max:.3f})")
    eps = 1e-6
    C = float(np.clip(C, reach_min + eps, reach_max - eps))

    # Knee from law of cosines (interior angle between thigh and shank).
    cos_knee = (A * A + B * B - C * C) / (2 * A * B)
    cos_knee = float(np.clip(cos_knee, -1.0, 1.0))
    knee_interior = np.arccos(cos_knee)
    q3 = lp.knee_forward * (np.pi - knee_interior)

    # Ankle roll & pitch: orient the shank so the ankle points at the hip.
    rx, ry, rz = r
    q5 = np.arctan2(ry, rz)                       # ankle roll
    # angle the thigh makes at the ankle (offset of hip direction from shank)
    alpha = np.arcsin(np.clip(A * np.sin(knee_interior) / C, -1.0, 1.0))
    q4 = -np.arctan2(rx, np.sign(rz + (rz == 0)) *
                     np.hypot(ry, rz)) - lp.knee_forward * alpha

    # Hip yaw/roll/pitch: remaining rotation R_hip^T R_foot = Rz Rx Ry (ankle)^-1
    R_ankle = Ry(q4) @ Rx(q5)
    R_hip_chain = hip_rot.T @ foot_rot @ R_ankle.T @ Ry(q3).T
    # R_hip_chain == Rz(q0) Rx(q1) Ry(q2). Decompose (ZXY intrinsic).
    # Rz(q0)Rx(q1)Ry(q2):
    #   [ c0c2 - s0s1s2 , -s0c1 , c0s2 + s0s1c2 ]
    #   [ s0c2 + c0s1s2 ,  c0c1 , s0s2 - c0s1c2 ]
    #   [ -c1s2         ,   s1  ,  c1c2         ]
    M = R_hip_chain
    q1h = np.arcsin(np.clip(M[2, 1], -1.0, 1.0))          # hip roll
    q0h = np.arctan2(-M[0, 1], M[1, 1])                   # hip yaw
    q2h = np.arctan2(-M[2, 0], M[2, 2])                   # hip pitch
    return np.array([q0h, q1h, q2h, q3, q4, q5])


# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    lp = LegParams()

    max_pos_err = 0.0
    max_rot_err = 0.0
    n_ok = 0
    n_trials = 3000
    for _ in range(n_trials):
        # random reachable joint config, IK it back, compare FK.
        q_true = np.array([
            rng.uniform(-0.4, 0.4),    # hip yaw
            rng.uniform(-0.3, 0.3),    # hip roll
            rng.uniform(-0.8, 0.4),    # hip pitch
            rng.uniform(0.05, 1.4) * lp.knee_forward,  # knee
            rng.uniform(-0.6, 0.6),    # ankle pitch
            rng.uniform(-0.3, 0.3),    # ankle roll
        ])
        hip_pos = np.array([0.0, lp.hip_offset_y, 0.9])
        hip_rot = Rz(rng.uniform(-0.2, 0.2))
        foot_pos, foot_rot = leg_fk(q_true, hip_pos, hip_rot, lp)
        try:
            q_sol = leg_ik(foot_pos, foot_rot, hip_pos, hip_rot, lp)
        except ValueError:
            continue
        fp, fr = leg_fk(q_sol, hip_pos, hip_rot, lp)
        pe = np.linalg.norm(fp - foot_pos)
        re = np.linalg.norm(fr - foot_rot)
        max_pos_err = max(max_pos_err, pe)
        max_rot_err = max(max_rot_err, re)
        n_ok += 1

    print(f"round-trip over {n_ok}/{n_trials} reachable samples")
    print(f"  max foot POSITION error: {max_pos_err*1e6:.3f} um")
    print(f"  max foot ROTATION error: {max_rot_err:.2e}")
    ok = max_pos_err < 1e-6 and max_rot_err < 1e-6
    print("IK round-trip:", "PASS" if ok else "FAIL")
