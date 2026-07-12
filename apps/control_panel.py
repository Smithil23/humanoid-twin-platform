#!/usr/bin/env python3
"""Native desktop control panel for the humanoid twin.

Pure Python, no browser, no server: a Tkinter window with joint sliders
and preset buttons, plus the MuJoCo 3D viewer, both driven by one live
physics loop.

    python apps/control_panel.py                       # panel + 3D viewer
    python apps/control_panel.py --config configs/star1.yaml --no-viewer

Safety: slider targets are slew-rate limited (max_speed rad/s), so the
robot moves smoothly instead of jerking - dragging a slider fast no
longer kicks the body over. Legs remain genuinely destabilizing to move
(nothing balances the robot yet); Reset recovers.
"""
from __future__ import annotations

import argparse
import threading
import time

import mujoco

from htp import PlatformConfig, Simulator
from htp.trajectory import squat_targets

GROUPS = {
    "Left arm":  ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                  "left_arm_yaw_joint", "left_elbow_pitch_joint",
                  "left_elbow_yaw_joint", "left_wrist_pitch_joint",
                  "left_wrist_roll_joint"],
    "Right arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                  "right_arm_yaw_joint", "right_elbow_pitch_joint",
                  "right_elbow_yaw_joint", "right_wrist_pitch_joint",
                  "right_wrist_roll_joint"],
    "Head":      ["neck_yaw_joint", "neck_pitch_joint"],
    "Waist":     ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"],
    "Legs (!)":  ["left_hip_pitch_joint", "left_knee_joint",
                  "left_ankle_pitch_joint", "right_hip_pitch_joint",
                  "right_knee_joint", "right_ankle_pitch_joint"],
}


class TwinController:
    """Physics loop with slew-rate-limited targets. GUI-independent."""

    def __init__(self, config: str, viewer: bool = True,
                 max_speed: float = 1.5):
        self.sim = Simulator(PlatformConfig.load(config))
        self.max_speed = max_speed          # rad/s toward desired targets
        self.desired = dict(self.sim.cfg.poses.stand)
        self.mode = "manual"
        self._squat_t0 = 0.0
        self._lock = threading.Lock()
        self._run = True
        self._use_viewer = viewer
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------ API
    def set_target(self, joint: str, value: float) -> None:
        with self._lock:
            self.mode = "manual"
            self.desired[joint] = float(value)

    def squat(self) -> None:
        with self._lock:
            self.mode = "squat"
            self._squat_t0 = self.sim.data.time

    def reset(self) -> None:
        with self._lock:
            self.sim.reset()
            self.desired = dict(self.sim.cfg.poses.stand)
            self.mode = "manual"

    def stop(self) -> None:
        self._run = False

    def status(self) -> dict:
        s = self.sim.state()
        return {"t": s.time, "z": s.base_height, "fz": s.contact_force_z,
                "up": self.sim.upright, "mode": self.mode}

    def joint_info(self, name: str) -> tuple[float, float, float]:
        m = self.sim.model
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
        if lo == hi:
            lo, hi = -3.14, 3.14
        return lo, hi, float(self.desired.get(name, 0.0))

    # ----------------------------------------------------------- loop
    def _step_targets(self, dt: float) -> None:
        """Glide actuator commands toward desired at max_speed."""
        max_d = self.max_speed * dt
        d = self.sim.data
        for name, want in self.desired.items():
            a = self.sim.act_index[f"{name}_act"]
            cur = d.ctrl[a]
            d.ctrl[a] = cur + max(-max_d, min(max_d, want - cur))

    def _loop(self) -> None:
        dt = self.sim.cfg.sim.timestep
        viewer_cm = (
            mujoco.viewer.launch_passive(self.sim.model, self.sim.data)
            if self._use_viewer else None
        )
        try:
            while self._run and (viewer_cm is None or
                                 viewer_cm.is_running()):
                t0 = time.time()
                with self._lock:
                    if self.mode == "squat":
                        tq = self.sim.data.time - self._squat_t0
                        self.desired.update(squat_targets(
                            tq, period=4.0, depth=1.0, start=0.5,
                            base=self.sim.cfg.poses.stand))
                        if tq > 5.0:
                            self.mode = "manual"
                    self._step_targets(dt * 5)
                    self.sim.step(5)
                if viewer_cm is not None:
                    viewer_cm.sync()
                left = dt * 5 - (time.time() - t0)
                if left > 0:
                    time.sleep(left)
        finally:
            if viewer_cm is not None:
                viewer_cm.close()


def build_gui(ctl: TwinController):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Humanoid Twin - control panel")
    root.geometry("420x560")

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    ttk.Button(top, text="Reset", command=ctl.reset).pack(
        side="left", padx=4)
    ttk.Button(top, text="Squat", command=ctl.squat).pack(
        side="left", padx=4)
    status = ttk.Label(top, text="...", font=("Consolas", 9))
    status.pack(side="right")

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=8)
    for group, joints in GROUPS.items():
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text=group)
        for name in joints:
            lo, hi, cur = ctl.joint_info(name)
            row = ttk.Frame(tab)
            row.pack(fill="x", pady=3)
            short = name.replace("_joint", "").replace("_", " ")
            val = ttk.Label(row, text=f"{cur:+.2f}", width=6,
                            font=("Consolas", 9))

            def on_move(v, n=name, lbl=val):
                lbl.config(text=f"{float(v):+.2f}")
                ctl.set_target(n, float(v))

            ttk.Label(row, text=short, width=18).pack(side="left")
            ttk.Scale(row, from_=lo, to=hi, value=cur,
                      command=on_move).pack(
                side="left", fill="x", expand=True, padx=6)
            val.pack(side="right")

    def tick() -> None:
        s = ctl.status()
        up = "upright" if s["up"] else "FALLEN - press Reset"
        status.config(
            text=f't={s["t"]:6.1f}s  z={s["z"]:.3f}m  {up}')
        root.after(150, tick)

    tick()
    root.protocol("WM_DELETE_WINDOW",
                  lambda: (ctl.stop(), root.destroy()))
    return root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/star1_visual.yaml")
    ap.add_argument("--no-viewer", action="store_true",
                    help="control panel only, no 3D window")
    ap.add_argument("--max-speed", type=float, default=1.5,
                    help="target slew rate [rad/s]")
    args = ap.parse_args()

    ctl = TwinController(args.config, viewer=not args.no_viewer,
                         max_speed=args.max_speed)
    build_gui(ctl).mainloop()


if __name__ == "__main__":
    main()
