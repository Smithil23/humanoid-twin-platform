"""
mj_ik.py — MuJoCo damped-least-squares leg IK.

Solves leg joint angles that place each foot SOLE at a desired world pose,
using the robot's EXACT kinematics (the live MjModel). This replaces the
analytic 2-link IK, which could not match STAR1's real geometry (offset hip
axes, ankle/sole offsets) and left the CoM behind the feet.

Method (per leg, damped least squares):
    err   = [sole_pos_err (3); sole_rot_err (3)]
    dq    = Jᵀ (J Jᵀ + λ²I)⁻¹ err
with the base fixed at the desired pelvis pose, so each leg's 6 joints solve
its own foot independently. An optional outer loop shifts the pelvis so the
whole-body CoM matches a target (handles the pose-dependent CoM offset).
"""
from __future__ import annotations

import mujoco
import numpy as np


def _quat_of_mat(mat3):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat3.reshape(9))
    return q


def rot_error(cur_mat, tgt_mat):
    """3-vector rotation error taking cur -> tgt (world frame)."""
    qc = _quat_of_mat(cur_mat)
    qt = _quat_of_mat(tgt_mat)
    err = np.zeros(3)
    mujoco.mju_subQuat(err, qt, qc)   # rotation from qc to qt
    return err


class LegIK:
    """Damped-least-squares foot-placement IK on the live model.

    foot_bodies: {"left": body_name, "right": body_name}   (ankle_roll link)
    leg_joints:  {"left": [6 joint names], "right": [...]}  (hip_yaw..ankle_roll)
    sole_offset: (3,) sole-contact point in the foot-body frame
                 (e.g. box offset + [0,0,-half_thickness]).
    """

    def __init__(self, model, foot_bodies, leg_joints, sole_offset):
        self.model = model
        self.d = mujoco.MjData(model)
        self.sole_offset = np.asarray(sole_offset, float)
        self.foot_bid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                         for s, b in foot_bodies.items()}
        self.dof, self.qadr, self.jrange = {}, {}, {}
        for s, names in leg_joints.items():
            jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                    for n in names]
            self.dof[s] = np.array([model.jnt_dofadr[j] for j in jids])
            self.qadr[s] = np.array([model.jnt_qposadr[j] for j in jids])
            self.jrange[s] = np.array([model.jnt_range[j] for j in jids])

    def _sole_pose(self, side):
        bid = self.foot_bid[side]
        R = self.d.xmat[bid].reshape(3, 3)
        p = self.d.xpos[bid] + R @ self.sole_offset
        return p, R

    def solve(self, qpos_init, pelvis_pos, pelvis_quat, sole_targets,
              iters=60, tol_pos=1e-3, tol_rot=np.radians(0.3),
              damping=1e-2, max_step=0.3):
        """Return ({side: (6,) joint angles}, info). Does not touch live sim.

        sole_targets: {side: (pos(3), mat(3x3))} desired world sole poses.
        """
        d = self.d
        d.qpos[:] = qpos_init
        d.qpos[0:3] = pelvis_pos
        d.qpos[3:7] = pelvis_quat
        mujoco.mj_kinematics(self.model, d); mujoco.mj_comPos(self.model, d)

        info = {}
        for side in sole_targets:
            tgt_p, tgt_R = sole_targets[side]
            dof = self.dof[side]
            qadr = self.qadr[side]
            jr = self.jrange[side]
            # Seed the knee (index 3) bent to its range midpoint so the DLS
            # starts well clear of the straight-leg singularity, where the
            # vertical Jacobian vanishes and the solver cannot raise the foot.
            knee_adr = qadr[3]
            d.qpos[knee_adr] = 0.5 * (jr[3, 0] + jr[3, 1])
            mujoco.mj_kinematics(self.model, d); mujoco.mj_comPos(self.model, d)
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            pos_err = rot_err = np.inf
            for it in range(iters):
                cur_p, cur_R = self._sole_pose(side)
                e_p = tgt_p - cur_p
                e_r = rot_error(cur_R, tgt_R)
                pos_err = float(np.linalg.norm(e_p))
                rot_err = float(np.linalg.norm(e_r))
                if pos_err < tol_pos and rot_err < tol_rot:
                    break
                # Jacobian of the sole point attached to the foot body.
                mujoco.mj_jac(self.model, d, jacp, jacr, cur_p,
                              self.foot_bid[side])
                J = np.vstack([jacp[:, dof], jacr[:, dof]])   # (6,6)
                e = np.concatenate([e_p, e_r])
                dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
                dq = np.clip(dq, -max_step, max_step)
                q = d.qpos[qadr] + dq
                q = np.clip(q, jr[:, 0], jr[:, 1])            # joint limits
                d.qpos[qadr] = q
                mujoco.mj_kinematics(self.model, d); mujoco.mj_comPos(self.model, d)
            info[side] = dict(pos_mm=pos_err * 1e3,
                              rot_deg=np.degrees(rot_err), iters=it + 1)
        angles = {s: d.qpos[self.qadr[s]].copy() for s in sole_targets}
        return angles, info

    def solve_with_com(self, qpos_init, pelvis_pos, pelvis_quat, sole_targets,
                       com_target_xy, com_iters=8, com_tol=2e-3, **kw):
        """Outer loop: shift the pelvis XY so the whole-body CoM lands on
        com_target_xy, re-solving the legs each time (handles pose-dependent
        CoM offset — issue #6)."""
        pp = np.array(pelvis_pos, float)
        last = None
        for c in range(com_iters):
            angles, info = self.solve(qpos_init, pp, pelvis_quat,
                                      sole_targets, **kw)
            # write solved legs into scratch, compute whole-body CoM
            mujoco.mj_forward(self.model, self.d)
            com = self.d.subtree_com[1][:2].copy()
            err = np.asarray(com_target_xy) - com
            last = (angles, info, com, float(np.linalg.norm(err)))
            if np.linalg.norm(err) < com_tol:
                break
            pp[0:2] += err                     # shift pelvis to move CoM
        angles, info, com, cerr = last
        info["com_xy"] = com
        info["com_err"] = cerr
        info["pelvis_xy"] = pp[0:2].copy()
        return angles, info
