#!/usr/bin/env python3
"""Twin Studio - native Python desktop app for the humanoid twin.

One window, one process: an embedded 3D view of the running physics
(MuJoCo offscreen renderer -> Qt image widget) with orbit/pan/zoom
camera, a control toolbar, and a live status bar. Built on PySide6.

    python apps/twin_studio.py
    python apps/twin_studio.py --config configs/star1.yaml   # no meshes

Camera: left-drag orbit, right-drag pan, wheel zoom.

Architecture (Phase A - panels dock in later phases):
    PhysicsEngine   background thread, steps MuJoCo, slew-limits targets
    Viewport3D      QWidget painting frames from mujoco.Renderer
    StudioWindow    QMainWindow: toolbar, central viewport, status bar
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import mujoco
import numpy as np
import PySide6  # noqa: F401  (imported before pyqtgraph on purpose)
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from htp import PlatformConfig, Simulator
from htp.balance import BalanceController
from htp.keyframes import KeyframePlayer, list_motions
from htp.trajectory import squat_targets


JOINT_GROUPS = {
    "Left arm":  ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                  "left_arm_yaw_joint", "left_elbow_pitch_joint",
                  "left_elbow_yaw_joint", "left_wrist_pitch_joint",
                  "left_wrist_roll_joint"],
    "Right arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                  "right_arm_yaw_joint", "right_elbow_pitch_joint",
                  "right_elbow_yaw_joint", "right_wrist_pitch_joint",
                  "right_wrist_roll_joint"],
    "Head":      ["neck_yaw_joint", "neck_pitch_joint"],
    "Waist":     ["waist_yaw_joint", "waist_pitch_joint",
                  "waist_roll_joint"],
    "Legs (!)":  ["left_hip_pitch_joint", "left_knee_joint",
                  "left_ankle_pitch_joint", "right_hip_pitch_joint",
                  "right_knee_joint", "right_ankle_pitch_joint"],
}

PRESETS = {
    "Right arm forward": {"right_shoulder_pitch_joint": -0.9},
    "T-pose": {"left_shoulder_roll_joint": 1.3,
               "right_shoulder_roll_joint": -1.3},
    "Look left": {"neck_yaw_joint": 0.9},
}


class PhysicsEngine:
    """Owns the sim thread. Slew-rate-limited targets, same as the
    Tkinter panel - a GUI never writes ctrl directly."""

    def __init__(self, config: str, max_speed: float = 1.5):
        self.sim = Simulator(PlatformConfig.load(config))
        self.max_speed = max_speed
        self.desired = dict(self.sim.cfg.poses.stand)
        self.mode = "manual"
        self._squat_t0 = 0.0
        self._player: KeyframePlayer | None = None
        self._motion_t0 = 0.0
        self.balance = BalanceController(self.sim)
        self.balance_on = False
        self.com_ref = (0.0, 0.0)
        self._push_until = 0.0
        self._push_force = 0.0
        self.paused = False
        self.lock = threading.Lock()
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def set_target(self, joint: str, value: float) -> None:
        with self.lock:
            self.mode = "manual"
            self.desired[joint] = float(value)

    def squat(self) -> None:
        with self.lock:
            self.mode = "squat"
            self._squat_t0 = self.sim.data.time

    def apply_pose(self, pose: dict[str, float]) -> None:
        """Overlay a named pose on the stand pose (bulk targets)."""
        with self.lock:
            self.mode = "manual"
            self.desired = dict(self.sim.cfg.poses.stand)
            self.desired.update(pose)

    def play_motion(self, path) -> None:
        with self.lock:
            self._player = KeyframePlayer.from_file(
                path, base=self.sim.cfg.poses.stand)
            self._motion_t0 = self.sim.data.time
            self.mode = "motion"

    def stop_motion(self) -> None:
        with self.lock:
            if self.mode == "motion":
                self.mode = "manual"
                self.com_ref = (0.0, 0.0)

    def motion_progress(self) -> tuple[str, float] | None:
        with self.lock:
            if self.mode != "motion":
                return None
            t = self.sim.data.time - self._motion_t0
            return self._player.name, self._player.progress(t)

    def reset(self) -> None:
        with self.lock:
            self.sim.reset()
            self.desired = dict(self.sim.cfg.poses.stand)
            self.balance.reset()
            self.sim.data.xfrc_applied[1, :] = 0
            self._push_until = 0.0        # sim clock rewinds on reset;
            self._push_force = 0.0        # a stale deadline re-pushes
            self.com_ref = (0.0, 0.0)
            self.mode = "manual"

    def push(self, force_n: float = 160.0, duration: float = 0.15) -> None:
        """Shove the torso forward - the balance controller's exam."""
        with self.lock:
            self._push_force = force_n
            self._push_until = self.sim.data.time + duration

    def stop(self) -> None:
        self._run = False

    def status(self) -> dict:
        with self.lock:                    # never read sim mid-step
            s = self.sim.state()
            up = self.sim.upright
        return {"t": s.time, "z": s.base_height, "fz": s.contact_force_z,
                "ncon": s.n_contacts, "up": up,
                "mode": self.mode, "paused": self.paused}

    def _loop(self) -> None:
        dt = self.sim.cfg.sim.timestep
        while self._run:
            t0 = time.time()
            if not self.paused:
                with self.lock:
                    if self.mode == "squat":
                        tq = self.sim.data.time - self._squat_t0
                        self.desired.update(squat_targets(
                            tq, period=4.0, depth=1.0, start=0.5,
                            base=self.sim.cfg.poses.stand))
                        if tq > 5.0:
                            self.mode = "manual"
                    elif self.mode == "motion":
                        tm = self.sim.data.time - self._motion_t0
                        tgt = self._player.targets(tm)
                        # com_x / com_y are pattern tracks, not joints:
                        # they command the balance controller's target
                        self.com_ref = (tgt.pop("com_x", 0.0),
                                        tgt.pop("com_y", 0.0))
                        self.desired.update(tgt)
                        if self._player.finished(tm):
                            self.mode = "manual"
                            self.com_ref = (0.0, 0.0)
                    max_d = self.max_speed * dt * 5
                    d = self.sim.data
                    for name, want in self.desired.items():
                        a = self.sim.act_index[f"{name}_act"]
                        d.ctrl[a] += max(-max_d,
                                         min(max_d, want - d.ctrl[a]))
                    # inner loop: balance feedback and push window run at
                    # the full physics rate (feedback quality depends on it)
                    for _ in range(5):
                        if self.balance_on:
                            op, orr = self.balance.update(
                                dt, ref=self.com_ref)
                            for j in BalanceController.ANKLE_PITCH:
                                a = self.sim.act_index[f"{j}_act"]
                                d.ctrl[a] = self.desired.get(j, 0.0) + op
                            for j in BalanceController.ANKLE_ROLL:
                                a = self.sim.act_index[f"{j}_act"]
                                d.ctrl[a] = self.desired.get(j, 0.0) + orr
                        d.xfrc_applied[1, 0] = (
                            self._push_force
                            if d.time < self._push_until else 0.0)
                        self.sim.step(1)
            left = dt * 5 - (time.time() - t0)
            if left > 0:
                time.sleep(left)


class Viewport3D(QWidget):
    """Embedded MuJoCo view: renders offscreen, paints the pixels,
    and maps mouse input to an orbit camera."""

    def __init__(self, engine: PhysicsEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.cam = mujoco.MjvCamera()
        self.cam.azimuth, self.cam.elevation = 135.0, -12.0
        self.cam.distance = 3.0
        self.cam.lookat[:] = [0.0, 0.0, 0.9]
        self.vopt = mujoco.MjvOption()
        self.show_balance = False
        self._renderer: mujoco.Renderer | None = None
        self._frame: QImage | None = None
        self._last_pos = None
        self.setMinimumSize(480, 360)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)                     # ~30 fps

    # ------------------------------------------------------- rendering
    def _ensure_renderer(self) -> None:
        w = min(max(self.width(), 320), 1920)
        h = min(max(self.height(), 240), 1080)
        if (self._renderer is None or self._renderer.width != w
                or self._renderer.height != h):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(
                self.engine.sim.model, height=h, width=w)

    def _refresh(self) -> None:
        self._ensure_renderer()
        with self.engine.lock:
            self._renderer.update_scene(
                self.engine.sim.data, camera=self.cam,
                scene_option=self.vopt)
            if self.show_balance:
                self._draw_balance_overlay()
        px = self._renderer.render()              # (H, W, 3) uint8
        h, w, _ = px.shape
        self._frame = QImage(
            np.ascontiguousarray(px).data, w, h, 3 * w,
            QImage.Format.Format_RGB888).copy()
        self.update()

    def paintEvent(self, ev) -> None:
        if self._frame is not None:
            QPainter(self).drawImage(self.rect(), self._frame)

    # ---------------------------------------------------------- camera
    def mousePressEvent(self, ev) -> None:
        self._last_pos = ev.position()

    def mouseMoveEvent(self, ev) -> None:
        if self._last_pos is None:
            return
        d = ev.position() - self._last_pos
        self._last_pos = ev.position()
        if ev.buttons() & Qt.MouseButton.LeftButton:
            self.cam.azimuth -= 0.4 * d.x()
            self.cam.elevation = float(np.clip(
                self.cam.elevation - 0.4 * d.y(), -89, 89))
        elif ev.buttons() & Qt.MouseButton.RightButton:
            az = np.deg2rad(self.cam.azimuth)
            right = np.array([np.cos(az), np.sin(az), 0.0])
            scale = 0.002 * self.cam.distance
            self.cam.lookat[:] -= right * d.x() * scale
            self.cam.lookat[2] += d.y() * scale

    def wheelEvent(self, ev) -> None:
        self.cam.distance = float(np.clip(
            self.cam.distance * 0.999 ** ev.angleDelta().y(), 0.5, 12))

    def _add_geom(self) -> "mujoco.MjvGeom | None":
        scn = self._renderer.scene
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        scn.ngeom += 1
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_SPHERE, np.zeros(3),
            np.zeros(3), np.eye(3).flatten(),
            np.array([1, 1, 1, 1], dtype=np.float32))
        return g

    def _draw_balance_overlay(self) -> None:
        """CoM ground projection (dot) + support polygon (lines).
        Green while stable, red when the margin goes negative."""
        sim = self.engine.sim
        poly = sim.support_polygon()
        margin = sim.balance_margin()
        col = ([0.2, 0.9, 0.4, 0.9] if margin > 0.02 else
               [0.95, 0.75, 0.1, 0.9] if margin > 0 else
               [0.95, 0.25, 0.2, 0.9])
        col = np.array(col, dtype=np.float32)
        # polygon edges
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            g = self._add_geom()
            if g is None:
                return
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_LINE, 4,
                np.array([a[0], a[1], 0.006]),
                np.array([b[0], b[1], 0.006]))
            g.rgba[:] = col
        # CoM projection dot
        com = sim.data.subtree_com[1]
        g = self._add_geom()
        if g is not None:
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size[:] = [0.02, 0.02, 0.02]
            g.pos[:] = [com[0], com[1], 0.02]
            g.rgba[:] = col
        # vertical drop line from actual CoM to floor
        g = self._add_geom()
        if g is not None:
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_LINE, 2,
                np.array([com[0], com[1], com[2]]),
                np.array([com[0], com[1], 0.01]))
            g.rgba[:] = np.array([1, 1, 1, 0.5], dtype=np.float32)

    def toggle_contacts(self, on: bool) -> None:
        f = mujoco.mjtVisFlag
        self.vopt.flags[f.mjVIS_CONTACTPOINT] = on
        self.vopt.flags[f.mjVIS_CONTACTFORCE] = on


class SlidersDock(QDockWidget):
    """Joint sliders grouped in tabs. Writes engine targets; can re-sync
    itself after resets/presets so knobs match reality."""

    def __init__(self, engine: PhysicsEngine, parent=None):
        super().__init__("Joints", parent)
        self.engine = engine
        self._rows: dict[str, tuple[QSlider, QLabel]] = {}
        tabs = QTabWidget()
        for group, joints in JOINT_GROUPS.items():
            page = QWidget()
            grid = QGridLayout(page)
            grid.setColumnStretch(1, 1)
            for r, name in enumerate(joints):
                m = engine.sim.model
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
                if lo == hi:
                    lo, hi = -3.14, 3.14
                cur = float(engine.desired.get(name, 0.0))
                short = name.replace("_joint", "").replace("_", " ")
                sld = QSlider(Qt.Orientation.Horizontal)
                sld.setRange(int(lo * 100), int(hi * 100))
                sld.setValue(int(cur * 100))
                val = QLabel(f"{cur:+.2f}")
                val.setMinimumWidth(48)
                sld.valueChanged.connect(
                    lambda v, n=name, lb=val: self._moved(n, v / 100, lb))
                grid.addWidget(QLabel(short), r, 0)
                grid.addWidget(sld, r, 1)
                grid.addWidget(val, r, 2)
                self._rows[name] = (sld, val)
            grid.setRowStretch(len(joints), 1)
            tabs.addTab(page, group)
        self.setWidget(tabs)

    def _moved(self, name: str, value: float, label: QLabel) -> None:
        label.setText(f"{value:+.2f}")
        self.engine.set_target(name, value)

    def sync(self) -> None:
        for name, (sld, val) in self._rows.items():
            want = float(self.engine.desired.get(name, 0.0))
            sld.blockSignals(True)
            sld.setValue(int(want * 100))
            sld.blockSignals(False)
            val.setText(f"{want:+.2f}")


class PlotsDock(QDockWidget):
    """Scrolling live plots: base height and one selectable joint torque."""

    WINDOW = 400          # samples kept (~40 s at 10 Hz)

    def __init__(self, engine: PhysicsEngine, parent=None):
        super().__init__("Telemetry", parent)
        self.engine = engine
        box = QWidget()
        lay = QVBoxLayout(box)

        self.sel = QComboBox()
        self.sel.addItems(engine.sim.hinge_names)
        self.sel.setCurrentText("left_knee_joint")
        lay.addWidget(self.sel)

        pg.setConfigOptions(antialias=True)
        self.p1 = pg.PlotWidget(title="base height [m]")
        self.p2 = pg.PlotWidget(title="joint torque [Nm]")
        for p in (self.p1, self.p2):
            p.showGrid(x=True, y=True, alpha=0.2)
            lay.addWidget(p)
        self.c1 = self.p1.plot(pen=pg.mkPen("#39c6b4", width=2))
        self.c2 = self.p2.plot(pen=pg.mkPen("#e0a63c", width=2))
        self.t, self.z, self.tau = [], [], []
        self.setWidget(box)

        timer = QTimer(self)
        timer.timeout.connect(self._sample)
        timer.start(100)

    def _sample(self) -> None:
        eng = self.engine
        with eng.lock:
            t = float(eng.sim.data.time)
            z = float(eng.sim.data.qpos[2])
            a = eng.sim.act_index[f"{self.sel.currentText()}_act"]
            tau = float(eng.sim.data.actuator_force[a])
        for buf, v in ((self.t, t), (self.z, z), (self.tau, tau)):
            buf.append(v)
            if len(buf) > self.WINDOW:
                buf.pop(0)
        self.c1.setData(self.t, self.z)
        self.c2.setData(self.t, self.tau)


class FeetDock(QDockWidget):
    """Per-foot ground reaction bars."""

    def __init__(self, engine: PhysicsEngine, parent=None):
        super().__init__("Ground reaction", parent)
        self.engine = engine
        box = QWidget()
        lay = QGridLayout(box)
        self.bars: dict[str, QProgressBar] = {}
        for col, link in enumerate(engine.sim.cfg.feet.links):
            side = "Left" if "left" in link else "Right"
            bar = QProgressBar()
            bar.setRange(0, 800)
            bar.setFormat("%v N")
            lay.addWidget(QLabel(side), 0, col)
            lay.addWidget(bar, 1, col)
            self.bars[link] = bar
        lay.setRowStretch(2, 1)
        self.setWidget(box)
        timer = QTimer(self)
        timer.timeout.connect(self._sample)
        timer.start(150)

    def _sample(self) -> None:
        with self.engine.lock:
            ff = self.engine.sim.foot_forces()
        for link, bar in self.bars.items():
            bar.setValue(int(min(ff[link], 800)))


class MotionDock(QDockWidget):
    """Motion library: pick a keyframe motion, play/stop, watch progress."""

    def __init__(self, engine: PhysicsEngine, window, parent=None):
        super().__init__("Motion", parent)
        self.engine = engine
        self.window = window
        self.motions = list_motions()

        box = QWidget()
        lay = QVBoxLayout(box)
        self.sel = QComboBox()
        self.sel.addItems(list(self.motions.keys()))
        lay.addWidget(self.sel)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        b_play = QPushButton("Play")
        b_stop = QPushButton("Stop")
        b_play.clicked.connect(self._play)
        b_stop.clicked.connect(self._stop)
        h.addWidget(b_play)
        h.addWidget(b_stop)
        lay.addWidget(row)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setFormat("idle")
        lay.addWidget(self.bar)
        lay.addStretch(1)
        self.setWidget(box)

        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(100)

    def _play(self) -> None:
        name = self.sel.currentText()
        if name not in self.motions:
            return
        self.engine.play_motion(self.motions[name])
        # motions with CoM tracks need the balance controller running
        with self.engine.lock:
            needs_balance = any(
                k.startswith("com_") for k in self.engine._player.tracks)
        if needs_balance and not self.engine.balance_on:
            self.engine.balance.reset()
            self.engine.balance_on = True
            self.window.logdock.log(
                "balance controller ENABLED (required by motion)")
        self.window.logdock.log(f"motion: {name}")
        self.window.sliders.sync()

    def _stop(self) -> None:
        self.engine.stop_motion()
        self.window.sliders.sync()

    def _tick(self) -> None:
        p = self.engine.motion_progress()
        if p is None:
            self.bar.setValue(0)
            self.bar.setFormat("idle")
        else:
            name, frac = p
            self.bar.setValue(int(frac * 100))
            self.bar.setFormat(f"{name}  %p%")


class LogDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Log", parent)
        self.box = QPlainTextEdit()
        self.box.setReadOnly(True)
        self.box.setMaximumBlockCount(500)
        self.setWidget(self.box)

    def log(self, msg: str) -> None:
        self.box.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")


class StudioWindow(QMainWindow):
    def __init__(self, engine: PhysicsEngine):
        super().__init__()
        self.engine = engine
        self.setWindowTitle("Humanoid Twin Studio")
        self.resize(1100, 720)

        self.view = Viewport3D(engine)
        self.setCentralWidget(self.view)

        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_pause = QAction("Pause", self)
        self.act_pause.setCheckable(True)
        self.act_pause.toggled.connect(self._toggle_pause)
        tb.addAction(self.act_pause)

        act_reset = QAction("Reset", self)
        act_reset.triggered.connect(self._do_reset)
        tb.addAction(act_reset)

        act_squat = QAction("Squat", self)
        act_squat.triggered.connect(self._do_squat)
        tb.addAction(act_squat)

        tb.addSeparator()
        cb = QCheckBox("Contact forces")
        cb.toggled.connect(self.view.toggle_contacts)
        tb.addWidget(cb)

        cb2 = QCheckBox("Balance")
        cb2.toggled.connect(
            lambda on: setattr(self.view, "show_balance", on))
        tb.addWidget(cb2)

        cb3 = QCheckBox("Balance ctrl")
        cb3.toggled.connect(self._toggle_balance)
        tb.addWidget(cb3)

        act_push = QAction("Push", self)
        act_push.triggered.connect(self._do_push)
        tb.addAction(act_push)

        tb.addSeparator()
        tb.addWidget(QLabel(" Preset: "))
        self.preset_box = QComboBox()
        self.preset_box.addItems(["Stand", *PRESETS.keys()])
        self.preset_box.activated.connect(self._apply_preset)
        tb.addWidget(self.preset_box)

        # docks
        self.sliders = SlidersDock(engine, self)
        self.plots = PlotsDock(engine, self)
        self.feet = FeetDock(engine, self)
        self.logdock = LogDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self.sliders)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self.plots)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.feet)
        self.motion = MotionDock(engine, self, self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                           self.motion)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea,
                           self.logdock)
        self.logdock.log("Twin Studio started")
        self._was_up = True

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(150)

    def _apply_preset(self) -> None:
        name = self.preset_box.currentText()
        self.engine.apply_pose(PRESETS.get(name, {}))
        self.sliders.sync()
        self.logdock.log(f"preset: {name}")

    def _toggle_balance(self, on: bool) -> None:
        self.engine.balance.reset()
        self.engine.balance_on = on
        self.logdock.log(
            f"balance controller {'ENABLED' if on else 'disabled'}")

    def _do_push(self) -> None:
        self.engine.push(160.0)
        self.logdock.log("push: 160 N forward, 0.15 s")

    def _do_reset(self) -> None:
        self.engine.reset()
        self.sliders.sync()
        self.logdock.log("reset to stand pose")

    def _do_squat(self) -> None:
        self.engine.squat()
        self.logdock.log("squat cycle started")

    def _toggle_pause(self, on: bool) -> None:
        self.engine.paused = on
        self.act_pause.setText("Play" if on else "Pause")

    def _update_status(self) -> None:
        s = self.engine.status()
        with self.engine.lock:
            margin = self.engine.sim.balance_margin()
        up = "upright" if s["up"] else "FALLEN - press Reset"
        self.statusBar().showMessage(
            f't = {s["t"]:7.2f} s    base z = {s["z"]:.3f} m    '
            f'ground = {s["fz"]:5.0f} N    contacts = {s["ncon"]}    '
            f'margin = {margin * 100:+.1f} cm    {s["mode"]}    {up}')
        if self._was_up and not s["up"]:
            self.logdock.log("WARNING: robot fell - press Reset")
        self._was_up = s["up"]

    def closeEvent(self, ev) -> None:
        self.engine.stop()
        super().closeEvent(ev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/star1_visual.yaml")
    ap.add_argument("--max-speed", type=float, default=1.5)
    args = ap.parse_args()

    app = QApplication(sys.argv)
    engine = PhysicsEngine(args.config, max_speed=args.max_speed)
    win = StudioWindow(engine)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
