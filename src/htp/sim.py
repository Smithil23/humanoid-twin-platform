"""Thin, typed wrapper around a MuJoCo model built by the pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import PlatformConfig
from .pipeline import UrdfPipeline


@dataclass
class SimState:
    """Snapshot of the quantities the platform exposes."""

    time: float
    base_height: float
    base_quat_w: float
    com: np.ndarray                 # (3,) whole-robot COM in world
    joint_pos: np.ndarray           # (n_hinge,)
    actuator_torque: np.ndarray     # (n_act,)
    contact_force_z: float          # total vertical ground reaction
    n_contacts: int
    qpos: np.ndarray                # full generalized position (base + joints)
    qvel: np.ndarray                # full generalized velocity


class Simulator:
    """Owns the MjModel/MjData pair and the control interface."""

    def __init__(self, cfg: PlatformConfig):
        self.cfg = cfg
        mjcf = UrdfPipeline(cfg).build()
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)
        self.hinge_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        self.act_index = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a): a
            for a in range(self.model.nu)
        }
        self.reset()

    # ------------------------------------------------------------ control
    def reset(self) -> None:
        """Start in the configured stand pose, soles just above the floor.

        The pose comes from config (many robots, STAR1 included, define
        zero as a crouch, with a separate upright pose). Joints are
        preset to the pose AND commanded to it, so the robot starts
        upright instead of straightening after the drop.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        for name, q in self.cfg.poses.stand.items():
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j >= 0:
                self.data.qpos[self.model.jnt_qposadr[j]] = q
            self.data.ctrl[self.act_index[f"{name}_act"]] = q
        mujoco.mj_forward(self.model, self.data)
        foot = self.cfg.feet.links[0]
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot)
        sole_z = self.data.xpos[bid][2] - self.cfg.feet.sole_drop
        self.data.qpos[2] += -sole_z + self.cfg.sim.settle_clearance
        mujoco.mj_forward(self.model, self.data)

    def set_joint_targets(self, targets: dict[str, float]) -> None:
        """Command target angles [rad] for named joints."""
        for name, value in targets.items():
            self.data.ctrl[self.act_index[f"{name}_act"]] = value

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------ sensing
    def state(self) -> SimState:
        d, m = self.data, self.model
        fz = 0.0
        for c in range(d.ncon):
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, c, f6)
            # contact frame normal is the first row of the frame
            fz += abs(f6[0])
        hinge_pos = np.array(
            [d.qpos[m.jnt_qposadr[j]] for j in range(m.njnt)
             if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        )
        return SimState(
            time=d.time,
            base_height=float(d.qpos[2]),
            base_quat_w=float(d.qpos[3]),
            com=d.subtree_com[1].copy(),
            joint_pos=hinge_pos,
            actuator_torque=d.actuator_force.copy(),
            contact_force_z=float(fz),
            n_contacts=int(d.ncon),
            qpos=d.qpos.copy(),
            qvel=d.qvel.copy(),
        )

    def foot_forces(self) -> dict[str, float]:
        """Vertical ground-reaction force per foot link [N]."""
        m, d = self.model, self.data
        want = {}
        for link in self.cfg.feet.links:
            want[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)] = link
        out = {link: 0.0 for link in self.cfg.feet.links}
        f6 = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            for g in (con.geom1, con.geom2):
                b = int(m.geom_bodyid[g])
                if b in want:
                    mujoco.mj_contactForce(m, d, c, f6)
                    out[want[b]] += abs(float(f6[0]))
        return out

    def support_polygon(self, min_force: float = 30.0) -> np.ndarray:
        """Convex hull (world XY) of the sole corners of LOADED feet.

        A foot carrying less than ``min_force`` [N] is airborne (or
        nearly) and does not contribute support - during a single-leg
        stand the polygon correctly collapses to the stance foot.
        Falls back to all feet if none are loaded (mid-air/fallen).
        """
        forces = self.foot_forces()
        links = [k for k in self.cfg.feet.links if forces[k] >= min_force]
        if not links:
            links = list(self.cfg.feet.links)
        f = self.cfg.feet
        hx, hy, hz = f.size[0] / 2, f.size[1] / 2, f.size[2] / 2
        ox, oy, oz = f.offset
        pts = []
        for link in links:
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, link)
            R = self.data.xmat[bid].reshape(3, 3)
            p0 = self.data.xpos[bid]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    local = np.array([ox + sx * hx, oy + sy * hy, oz - hz])
                    pts.append((p0 + R @ local)[:2])
        pts = np.array(pts)
        # convex hull, gift-wrapping (tiny N, no scipy needed)
        hull = []
        start = int(np.argmin(pts[:, 0]))
        i = start
        while True:
            hull.append(i)
            j = (i + 1) % len(pts)
            for k in range(len(pts)):
                c = np.cross(pts[j] - pts[i], pts[k] - pts[i])
                if c < 0:
                    j = k
            i = j
            if i == start:
                break
        return pts[hull]

    def balance_margin(self) -> float:
        """Signed distance [m] from the CoM ground projection to the
        support-polygon boundary. Positive inside (stable), negative
        outside (tipping)."""
        poly = self.support_polygon()
        com = self.data.subtree_com[1][:2]
        n = len(poly)
        inside = True
        dmin = np.inf
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            e = b - a
            if np.cross(e, com - a) < 0:
                inside = False
            t = np.clip(np.dot(com - a, e) / np.dot(e, e), 0, 1)
            dmin = min(dmin, float(np.linalg.norm(com - (a + t * e))))
        return dmin if inside else -dmin

    def body_xz(self, names: list[str]) -> list[tuple[float, float]]:
        """(x, z) world positions of named bodies - for the skeleton view."""
        out = []
        for n in names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            p = self.data.xpos[bid]
            out.append((float(p[0]), float(p[2])))
        return out

    @property
    def upright(self) -> bool:
        return abs(self.data.qpos[3]) > 0.95
