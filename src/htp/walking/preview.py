"""
preview.py — LIPM / ZMP preview controller (Kajita 2003, cart-table model).

Generates a CoM trajectory that tracks a desired ZMP reference using
optimal preview control. This is the feedforward heart of the walk; a
runtime ZMP/CoM feedback stabiliser (your capture-point / ankle stack)
sits on top to reject model error and disturbances.

Model (per horizontal axis, decoupled x and y):
    state  X = [c, c_dot, c_ddot]^T      (CoM position, vel, acc)
    input  u = c_dddot                    (CoM jerk)
    output p = c - (z_c/g) * c_ddot       (ZMP under cart-table assumption)

References:
    Kajita et al., "Biped Walking Pattern Generation by using Preview
    Control of Zero-Moment Point", ICRA 2003.
    Katayama et al., "Design of an optimal controller for a discrete-time
    system subject to previewable demand", Int. J. Control 1985.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are


class PreviewController:
    """Optimal ZMP-preview controller for one horizontal axis.

    The same controller object is reused for x and y (the LIPM decouples).
    """

    def __init__(
        self,
        dt: float,
        z_c: float,
        preview_horizon: float = 1.6,
        g: float = 9.81,
        Qe: float = 1.0,
        Qx: float = 0.0,
        R: float = 1.0e-6,
    ) -> None:
        self.dt = dt
        self.z_c = z_c
        self.g = g
        self.n_preview = int(round(preview_horizon / dt))

        # --- Discrete triple-integrator (jerk input) -------------------
        A = np.array([[1.0, dt, dt * dt / 2.0],
                      [0.0, 1.0, dt],
                      [0.0, 0.0, 1.0]])
        B = np.array([[dt ** 3 / 6.0],
                      [dt ** 2 / 2.0],
                      [dt]])
        C = np.array([[1.0, 0.0, -z_c / g]])
        self.A, self.B, self.C = A, B, C

        # --- Augmented system for tracking (Katayama) ------------------
        # X_tilde = [e_k ; dx_k],  e_k = y_k - y_ref_k,  dx_k = x_k - x_{k-1}
        CA = C @ A                      # (1,3)
        CB = C @ B                      # (1,1)
        A_t = np.block([[np.ones((1, 1)), CA],
                        [np.zeros((3, 1)), A]])          # (4,4)
        B_t = np.block([[CB],
                        [B]])                            # (4,1)
        I_t = np.block([[np.ones((1, 1))],
                        [np.zeros((3, 1))]])             # (4,1) ref channel

        Q_t = np.zeros((4, 4))
        Q_t[0, 0] = Qe
        Q_t[1:, 1:] = Qx * np.eye(3)
        R_t = np.array([[R]])

        P = solve_discrete_are(A_t, B_t, Q_t, R_t)

        denom = R_t + B_t.T @ P @ B_t                    # (1,1)
        denom_inv = np.linalg.inv(denom)

        # Feedback gains: [Gi | Gx]
        K = denom_inv @ (B_t.T @ P @ A_t)                # (1,4)
        self.Gi = float(K[0, 0])                         # integral gain
        self.Gx = K[0, 1:].reshape(1, 3)                 # state gain (1,3)

        # Preview gains Gd(l), l = 1..n_preview
        Ac = A_t - B_t @ K                                # closed-loop (4,4)
        Gd = np.zeros(self.n_preview)
        X = -Ac.T @ P @ I_t                              # (4,1)
        Gd[0] = -self.Gi
        for l in range(1, self.n_preview):
            Gd[l] = (denom_inv @ (B_t.T @ X)).item()
            X = Ac.T @ X
        self.Gd = Gd

        # Running controller state (per axis): x=[c,cdot,cddot], integral e
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, c0: float = 0.0) -> None:
        self.x = np.array([[c0], [0.0], [0.0]])
        self.sum_e = 0.0

    def step(self, zmp_ref_window: np.ndarray, zmp_ref_now: float) -> tuple:
        """Advance one control tick.

        Args:
            zmp_ref_window: future ZMP reference, length >= n_preview,
                            zmp_ref_window[0] = reference at k+1.
            zmp_ref_now:    desired ZMP at the current step k.

        Returns:
            (com_pos, com_vel, com_acc, zmp_out) for this axis this tick.
        """
        y = (self.C @ self.x).item()          # current ZMP output
        e = y - zmp_ref_now
        self.sum_e += e

        preview = float(self.Gd @ np.asarray(zmp_ref_window[: self.n_preview]))
        u = -self.Gi * self.sum_e - (self.Gx @ self.x).item() - preview

        self.x = self.A @ self.x + self.B * u
        c, cdot, cddot = self.x.flatten()
        zmp_out = (self.C @ self.x).item()
        return c, cdot, cddot, zmp_out


def generate_com_trajectory(pc: PreviewController,
                            zmp_ref: np.ndarray,
                            c0: float = 0.0) -> dict:
    """Offline: run the preview controller across a full ZMP reference.

    Args:
        pc:      a PreviewController (reused per axis).
        zmp_ref: (N,) desired ZMP samples at control rate.
        c0:      initial CoM position.

    Returns dict with 'com', 'com_vel', 'com_acc', 'zmp'.
    """
    N = len(zmp_ref)
    padded = np.concatenate([zmp_ref, np.repeat(zmp_ref[-1], pc.n_preview + 1)])
    pc.reset(c0)
    com = np.zeros(N)
    vel = np.zeros(N)
    acc = np.zeros(N)
    zmp = np.zeros(N)
    for k in range(N):
        window = padded[k + 1: k + 1 + pc.n_preview]
        c, cd, cdd, p = pc.step(window, padded[k])
        com[k], vel[k], acc[k], zmp[k] = c, cd, cdd, p
    return {"com": com, "com_vel": vel, "com_acc": acc, "zmp": zmp}


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Self-test: square-wave ZMP reference (lateral sway), check tracking.
    dt, z_c = 0.005, 0.80
    pc = PreviewController(dt=dt, z_c=z_c, preview_horizon=1.6)

    T = 8.0
    N = int(T / dt)
    t = np.arange(N) * dt
    # lateral ZMP hops between +/- 0.09 m every 0.7 s (like stepping)
    step_T = 0.7
    zmp_ref = 0.09 * (2 * ((t // step_T).astype(int) % 2) - 1)

    out = generate_com_trajectory(pc, zmp_ref, c0=zmp_ref[0])

    # Steady-state tracking error (ignore first preview horizon transient)
    warm = pc.n_preview
    err = out["zmp"][warm:] - zmp_ref[warm:]
    print(f"n_preview          = {pc.n_preview} samples "
          f"({pc.n_preview * dt:.2f} s)")
    print(f"Gi                 = {pc.Gi:.4f}")
    print(f"Gx                 = {pc.Gx.flatten()}")
    print(f"Gd[:5]             = {pc.Gd[:5]}")
    print(f"ZMP RMS track err  = {np.sqrt(np.mean(err**2))*1000:.3f} mm")
    print(f"ZMP max track err  = {np.max(np.abs(err))*1000:.3f} mm")
    print(f"CoM range          = [{out['com'].min():.3f}, "
          f"{out['com'].max():.3f}] m  (ref +/-0.09)")

    # Sanity: CoM should lead the ZMP (preview) and be smoother.
    print(f"CoM peak accel     = {np.max(np.abs(out['com_acc'])):.3f} m/s^2")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, zmp_ref, "k--", lw=1, label="ZMP ref")
        ax.plot(t, out["zmp"], "C3", lw=1, label="ZMP actual")
        ax.plot(t, out["com"], "C0", lw=1.5, label="CoM")
        ax.set_xlabel("time [s]"); ax.set_ylabel("lateral [m]")
        ax.legend(loc="upper right"); ax.grid(alpha=0.3)
        ax.set_title("LIPM preview control — lateral ZMP tracking")
        fig.tight_layout(); fig.savefig("preview_selftest.png", dpi=110)
        print("wrote preview_selftest.png")
    except Exception as exc:
        print("plot skipped:", exc)
