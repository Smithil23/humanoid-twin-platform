"""Live dashboard for the humanoid twin.

Runs the simulation in a background thread and streams state over a
websocket at ~30 Hz. Commands (reset / squat / joint targets) arrive on
the same socket.

    uvicorn apps.dashboard.server:app --reload
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import mujoco

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from htp import PlatformConfig, Simulator
from htp.trajectory import squat_targets

CONFIG = "configs/star1.yaml"
SKELETON = [
    "base_link",
    "left_hip_pitch_link", "left_knee_link", "left_ankle_roll_link",
    "base_link",
    "right_hip_pitch_link", "right_knee_link", "right_ankle_roll_link",
    "base_link",
    "neck_yaw_link",
]

app = FastAPI(title="Humanoid Twin Platform")


class TwinRunner:
    """Owns the sim thread; the websocket reads snapshots from it."""

    def __init__(self) -> None:
        self.sim = Simulator(PlatformConfig.load(CONFIG))
        self.mode = "stand"           # stand | squat
        self._squat_t0 = 0.0
        self._lock = threading.Lock()
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        dt = self.sim.cfg.sim.timestep
        import time as _time
        while self._run:
            with self._lock:
                if self.mode == "squat":
                    t = self.sim.data.time - self._squat_t0
                    self.sim.set_joint_targets(
                        squat_targets(t, start=0.5, base=self.sim.cfg.poses.stand)
                    )
                    if t > 5.0:
                        self.mode = "stand"
                self.sim.step(5)      # 10 ms of physics per outer tick
            _time.sleep(dt * 5)       # ~soft real-time

    # -------------------------------------------------------- commands
    def command(self, msg: dict) -> None:
        with self._lock:
            if msg.get("cmd") == "reset":
                self.sim.reset()
                self.mode = "stand"
            elif msg.get("cmd") == "squat":
                self._squat_t0 = self.sim.data.time
                self.mode = "squat"
            elif msg.get("cmd") == "target":
                self.sim.set_joint_targets(
                    {msg["joint"]: float(msg["value"])}
                )

    # ------------------------------------------------- joint directory
    GROUPS = {
        "left arm":  ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                      "left_arm_yaw_joint", "left_elbow_pitch_joint",
                      "left_elbow_yaw_joint", "left_wrist_pitch_joint",
                      "left_wrist_roll_joint"],
        "right arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                      "right_arm_yaw_joint", "right_elbow_pitch_joint",
                      "right_elbow_yaw_joint", "right_wrist_pitch_joint",
                      "right_wrist_roll_joint"],
        "head":      ["neck_yaw_joint", "neck_pitch_joint"],
        "waist":     ["waist_yaw_joint", "waist_pitch_joint",
                      "waist_roll_joint"],
        "legs":      ["left_hip_pitch_joint", "left_knee_joint",
                      "left_ankle_pitch_joint", "right_hip_pitch_joint",
                      "right_knee_joint", "right_ankle_pitch_joint"],
    }

    def joint_directory(self) -> dict:
        """Groups -> sliders: name, range, and current target."""
        m = self.sim.model
        out: dict[str, list] = {}
        with self._lock:
            for group, names in self.GROUPS.items():
                rows = []
                for n in names:
                    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
                    if j < 0:
                        continue
                    lo, hi = (float(m.jnt_range[j][0]),
                              float(m.jnt_range[j][1]))
                    if lo == hi:
                        lo, hi = -3.14, 3.14
                    a = self.sim.act_index[f"{n}_act"]
                    rows.append({
                        "name": n, "min": round(lo, 3), "max": round(hi, 3),
                        "value": round(float(self.sim.data.ctrl[a]), 3),
                    })
                out[group] = rows
        return out

    def snapshot(self) -> dict:
        with self._lock:
            s = self.sim.state()
            pts = self.sim.body_xz(SKELETON)
        return {
            "t": round(s.time, 3),
            "z": round(s.base_height, 4),
            "com": [round(v, 4) for v in s.com.tolist()],
            "fz": round(s.contact_force_z, 1),
            "ncon": s.n_contacts,
            "upright": bool(s.base_quat_w and abs(s.base_quat_w) > 0.95),
            "skel": pts,
            "mode": self.mode,
        }


runner: TwinRunner | None = None


@app.on_event("startup")
def _startup() -> None:
    global runner
    runner = TwinRunner()


@app.get("/joints")
def joints() -> dict:
    return runner.joint_directory()


@app.get("/")
def index() -> HTMLResponse:
    html = (Path(__file__).parent / "static" / "index.html").read_text()
    return HTMLResponse(html)


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()

    async def sender() -> None:
        while True:
            await sock.send_json(runner.snapshot())
            await asyncio.sleep(1 / 30)

    task = asyncio.create_task(sender())
    try:
        while True:
            msg = await sock.receive_json()
            runner.command(msg)
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
