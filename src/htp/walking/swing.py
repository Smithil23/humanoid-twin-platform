"""
swing.py — swing foot trajectory.

Interpolates the swinging foot from its liftoff pose to its next
touchdown pose during single support, with a smooth vertical clearance
profile and zero velocity/acceleration at both ends (so the foot lands
softly and lifts cleanly).

Horizontal: min-jerk (quintic) interpolation, x/y and yaw.
Vertical:   single-hump profile peaking at mid-swing, returning to the
            ground height at touchdown.
"""
from __future__ import annotations

import numpy as np


def _quintic(s: float) -> float:
    """Min-jerk scaling, s in [0,1] -> [0,1], zero vel/acc at ends."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _quintic_vel(s: float) -> float:
    return 30 * s**2 - 60 * s**3 + 30 * s**4


def swing_foot_pose(phase: float,
                    p_from: np.ndarray,
                    p_to: np.ndarray,
                    yaw_from: float,
                    yaw_to: float,
                    step_height: float,
                    ground_z: float = 0.0) -> tuple:
    """Foot pose partway through a swing.

    Args:
        phase:      0..1 progress through single support.
        p_from/to:  (x, y) liftoff / touchdown ground positions.
        yaw_from/to: foot yaw at liftoff / touchdown [rad].
        step_height: peak clearance above ground_z [m].

    Returns:
        (pos_xyz (3,), yaw, vel_xyz (3,)) — world frame.
    """
    s = float(np.clip(phase, 0.0, 1.0))
    a = _quintic(s)
    da = _quintic_vel(s)

    xy = p_from + a * (p_to - p_from)
    yaw = yaw_from + a * (yaw_to - yaw_from)

    # Vertical: raised cosine hump, peak at mid-swing, 0 at both ends.
    z = ground_z + step_height * 0.5 * (1.0 - np.cos(2.0 * np.pi * s))
    dz = step_height * 0.5 * (2.0 * np.pi) * np.sin(2.0 * np.pi * s)

    pos = np.array([xy[0], xy[1], z])
    vxy = (p_to - p_from) * da
    vel = np.array([vxy[0], vxy[1], dz])
    return pos, yaw, vel


# ----------------------------------------------------------------------
if __name__ == "__main__":
    p_from = np.array([0.0, 0.11])
    p_to = np.array([0.18, 0.11])
    N = 60
    traj = np.array([
        swing_foot_pose(k / (N - 1), p_from, p_to, 0.0, 0.0, 0.05)[0]
        for k in range(N)
    ])
    print(f"liftoff  z={traj[0,2]:.4f}  touchdown z={traj[-1,2]:.4f}")
    print(f"peak clearance z={traj[:,2].max():.4f} at "
          f"phase={traj[:,2].argmax()/(N-1):.2f}")
    print(f"forward travel x: {traj[0,0]:.3f} -> {traj[-1,0]:.3f}")
    assert abs(traj[0, 2]) < 1e-9 and abs(traj[-1, 2]) < 1e-9, "foot not grounded at ends"
    print("swing endpoints grounded: OK")
