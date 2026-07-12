"""URDF -> simulation-ready MJCF pipeline.

Takes any humanoid URDF and produces a MuJoCo scene that is ready to
simulate: floating base, ground plane, foot contact geometry, and
position-controlled actuators on every hinge joint.

Stages
------
1. sanitize   : strip mesh references (works without CAD assets)
2. float      : insert a floating joint  world -> root link
3. add_feet   : attach box collision geometry to the foot links
4. to_mjcf    : load with MuJoCo, round-trip to MJCF
5. add_scene  : ground plane, solver options
6. add_actors : per-joint position actuators with per-group gains
"""
from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from .config import PlatformConfig


class UrdfPipeline:
    """Builds a simulation-ready MJCF string from a URDF file."""

    def __init__(self, cfg: PlatformConfig):
        self.cfg = cfg

    # ------------------------------------------------------------- stages
    def sanitize(self, tree: ET.ElementTree) -> int:
        """Apply the mesh policy.

        mesh_mode "strip"  : remove every mesh reference (visual and
                             collision) - runs anywhere, no CAD needed.
        mesh_mode "visual" : keep visual meshes for 3D rendering, remove
                             only collision meshes (contact still comes
                             from the parametric sole boxes). Requires
                             the mesh files to exist at the referenced
                             paths. A <mujoco> extension element is added
                             so the importer keeps visual geometry.

        Inertial data is never touched, so dynamics are identical in
        both modes. Returns the number of removed elements.
        """
        mode = self.cfg.robot.mesh_mode
        removed = 0
        root = tree.getroot()
        tags = ("visual", "collision") if mode == "strip" else ("collision",)
        for link in root.iter("link"):
            for tag in tags:
                for el in list(link.findall(tag)):
                    if el.find(".//mesh") is not None:
                        link.remove(el)
                        removed += 1
        if mode == "visual":
            if self.cfg.robot.mesh_dir:
                self._rewrite_mesh_paths(root)
            ext = root.find("mujoco")
            if ext is None:
                ext = ET.SubElement(root, "mujoco")
            comp = ext.find("compiler")
            if comp is None:
                comp = ET.SubElement(ext, "compiler")
            comp.set("discardvisual", "false")
            comp.set("balanceinertia", "true")
            comp.set("strippath", "false")
        return removed

    def _rewrite_mesh_paths(self, root: ET.Element) -> None:
        """Point every mesh reference at cfg.robot.mesh_dir.

        URDFs exported from CAD tools often hardcode absolute paths
        from the author's machine; rewriting by basename makes the
        model portable (CI, Docker, other machines).
        """
        mesh_dir = Path(self.cfg.robot.mesh_dir).resolve()
        for mesh in root.iter("mesh"):
            fn = mesh.get("filename")
            if fn:
                base = fn.replace("\\", "/").rsplit("/", 1)[-1]
                mesh.set("filename", str(mesh_dir / base))

    def add_floating_base(self, tree: ET.ElementTree) -> None:
        root = tree.getroot()
        ET.SubElement(root, "link", {"name": "world"})
        j = ET.SubElement(
            root, "joint", {"name": "floating_base", "type": "floating"}
        )
        ET.SubElement(j, "parent", {"link": "world"})
        ET.SubElement(j, "child", {"link": self.cfg.robot.root_link})
        ET.SubElement(j, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

    def add_foot_geometry(self, tree: ET.ElementTree) -> None:
        f = self.cfg.feet
        for link in tree.getroot().iter("link"):
            if link.get("name") in f.links:
                col = ET.SubElement(link, "collision")
                ET.SubElement(
                    col, "origin",
                    {"xyz": " ".join(map(str, f.offset)), "rpy": "0 0 0"},
                )
                geo = ET.SubElement(col, "geometry")
                ET.SubElement(
                    geo, "box", {"size": " ".join(map(str, f.size))}
                )

    def to_mjcf(self, urdf_text: str) -> str:
        """Round-trip through MuJoCo's URDF importer to obtain MJCF."""
        with tempfile.TemporaryDirectory() as td:
            up = Path(td) / "robot.urdf"
            up.write_text(urdf_text)
            model = mujoco.MjModel.from_xml_path(str(up))
            xp = Path(td) / "robot.xml"
            mujoco.mj_saveLastXML(str(xp), model)
            return xp.read_text()

    def add_scene(self, mjcf: str) -> str:
        s = self.cfg.sim
        floor = (
            '<geom name="floor" type="plane" size="10 10 0.1" '
            f'rgba="0.85 0.85 0.85 1" friction="{s.friction} 0.005 0.0001"/>'
        )
        mjcf = mjcf.replace("<worldbody>", f"<worldbody>\n    {floor}", 1)
        opt = (
            f'  <option timestep="{s.timestep}" '
            f'integrator="{s.integrator}"/>\n'
            '  <visual><global offwidth="1920" offheight="1080"/></visual>\n'
            "</mujoco>"
        )
        return mjcf.replace("</mujoco>", opt)

    def add_actuators(self, mjcf: str) -> str:
        """Position actuator per hinge joint, gains resolved per group.

        Implicit position actuators remain stable at gains where an
        explicitly-applied PD torque would diverge.
        """
        with tempfile.TemporaryDirectory() as td:
            xp = Path(td) / "m.xml"
            xp.write_text(mjcf)
            model = mujoco.MjModel.from_xml_path(str(xp))

        lines = []
        for j in range(model.njnt):
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            g = self.cfg.gains.resolve(name)
            lines.append(
                f'<position name="{name}_act" joint="{name}" '
                f'kp="{g.kp}" kv="{g.kv}" '
                f'forcerange="-{g.torque_limit} {g.torque_limit}"/>'
            )
        block = "  <actuator>\n    " + "\n    ".join(lines) + "\n  </actuator>\n</mujoco>"
        return mjcf.replace("</mujoco>", block, 1)

    # -------------------------------------------------------------- build
    def build(self) -> str:
        """Run all stages and return the final MJCF string."""
        tree = ET.parse(self.cfg.robot.urdf)
        self.sanitize(tree)
        self.add_floating_base(tree)
        self.add_foot_geometry(tree)
        urdf_text = ET.tostring(tree.getroot(), encoding="unicode")
        mjcf = self.to_mjcf(urdf_text)
        mjcf = self.add_scene(mjcf)
        mjcf = self.add_actuators(mjcf)
        return mjcf

    def build_to(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.build())
        return path
