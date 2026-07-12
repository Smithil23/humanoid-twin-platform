#!/usr/bin/env python3
"""Cross-validate the MuJoCo twin against Pinocchio rigid-body dynamics.

Replaces the MATLAB validation step with a 100% Python pipeline:

    python scripts/run_squat.py --npz runs/squat.npz
    python scripts/validate_dynamics.py runs/squat.npz

For every logged sample, joint torques are recomputed with Pinocchio's
recursive Newton-Euler algorithm (inverse dynamics) on the same URDF,
using a floating base driven by the logged base motion. Torques for
contact-free joints (arms, neck) should closely match what the MuJoCo
actuators produced; leg joints differ by the ground-reaction
contribution, which is expected and physically meaningful.

Why this is a real check: Pinocchio and MuJoCo share no code - they are
two independent implementations of rigid-body dynamics. Agreement means
the twin's physics can be trusted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pinocchio as pin


def mujoco_to_pin_q(model: pin.Model, qpos: np.ndarray,
                    joint_cols: dict[str, int],
                    hinge_names: list[str]) -> np.ndarray:
    """Map a MuJoCo qpos row onto Pinocchio's configuration vector.

    MuJoCo free joint: [x y z, qw qx qy qz]; Pinocchio free-flyer:
    [x y z, qx qy qz qw]. Hinges are matched by joint name.
    """
    q = pin.neutral(model)
    q[0:3] = qpos[0:3]
    q[3:7] = [qpos[4], qpos[5], qpos[6], qpos[3]]     # wxyz -> xyzw
    for name, col in joint_cols.items():
        jid = model.getJointId(name)
        q[model.joints[jid].idx_q] = qpos[col]
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="runs/squat.npz from run_squat.py")
    ap.add_argument("--urdf", default="assets/star1_fixed.urdf")
    ap.add_argument("--joint", default="left_elbow_pitch_joint",
                    help="contact-free joint to plot/score")
    ap.add_argument("--plot", default="docs/validation_python.png")
    args = ap.parse_args()

    L = np.load(args.log, allow_pickle=True)
    hinge_names = [str(n) for n in L["joint_names"]]
    t, qpos_log, qvel_log = L["time"], L["qpos"], L["qvel"]
    tau_mj = L["actuator_torque"]

    model = pin.buildModelFromUrdf(args.urdf, pin.JointModelFreeFlyer())
    data = model.createData()

    # column of each named hinge in the MuJoCo qpos rows
    # (free joint occupies cols 0..6, hinges follow in model order)
    joint_cols = {n: 7 + i for i, n in enumerate(hinge_names)}

    N = len(t)
    tau_pin = np.zeros((N, len(hinge_names)))
    qs = [mujoco_to_pin_q(model, qpos_log[k], joint_cols, hinge_names)
          for k in range(N)]

    for k in range(1, N - 1):
        dt2 = t[k + 1] - t[k - 1]
        v = pin.difference(model, qs[k - 1], qs[k + 1]) / dt2
        v_prev = pin.difference(model, qs[k - 1], qs[k]) / (t[k] - t[k - 1])
        v_next = pin.difference(model, qs[k], qs[k + 1]) / (t[k + 1] - t[k])
        a = (v_next - v_prev) / (dt2 / 2.0)
        tau_full = pin.rnea(model, data, qs[k], v, a)
        for i, n in enumerate(hinge_names):
            jid = model.getJointId(n)
            tau_pin[k, i] = tau_full[model.joints[jid].idx_v]

    # score the requested contact-free joint (skip endpoint samples)
    j = hinge_names.index(args.joint)
    err = tau_mj[1:-1, j] - tau_pin[1:-1, j]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    span = float(np.ptp(tau_mj[1:-1, j])) or 1.0
    print(f"joint: {args.joint}")
    print(f"RMSE MuJoCo vs Pinocchio: {rmse:.4f} Nm "
          f"({100 * rmse / span:.1f}% of torque range)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t[1:-1], tau_mj[1:-1, j], lw=1.6, label="MuJoCo actuator")
        ax.plot(t[1:-1], tau_pin[1:-1, j], "--", lw=1.4,
                label="Pinocchio RNEA")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("torque [Nm]")
        ax.set_title(f"Twin cross-validation: {args.joint}")
        ax.legend()
        ax.grid(alpha=0.3)
        Path(args.plot).parent.mkdir(exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"plot saved: {args.plot}")
    except ImportError:
        print("matplotlib not installed - skipped plot "
              "(pip install matplotlib)")


if __name__ == "__main__":
    main()
