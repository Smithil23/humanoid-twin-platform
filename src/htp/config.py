"""Typed configuration, loaded from YAML.

Everything robot-specific lives in the config file, keeping the platform
code robot-agnostic: point ``robot.urdf`` at a different humanoid and
adjust the foot links / gain groups.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RobotCfg:
    urdf: str
    root_link: str = "base_link"
    mesh_mode: str = "strip"          # "strip" | "visual"
    mesh_dir: str | None = None       # rewrite mesh paths to this folder


@dataclass
class FeetCfg:
    links: list[str] = field(default_factory=list)
    size: list[float] = field(default_factory=lambda: [0.22, 0.10, 0.03])
    offset: list[float] = field(default_factory=lambda: [0.02, 0.0, -0.045])

    @property
    def sole_drop(self) -> float:
        """Distance from the foot link origin down to the sole."""
        return -self.offset[2] + self.size[2] / 2


@dataclass
class GainGroup:
    kp: float
    kv: float
    torque_limit: float


@dataclass
class GainsCfg:
    groups: dict[str, GainGroup] = field(default_factory=dict)
    patterns: dict[str, list[str]] = field(default_factory=dict)
    default: GainGroup = field(
        default_factory=lambda: GainGroup(kp=50, kv=4, torque_limit=30)
    )

    def resolve(self, joint_name: str) -> GainGroup:
        for group, pats in self.patterns.items():
            if any(p in joint_name for p in pats):
                return self.groups[group]
        return self.default


@dataclass
class PosesCfg:
    stand: dict[str, float] = field(default_factory=dict)


@dataclass
class SimCfg:
    timestep: float = 0.002
    integrator: str = "implicitfast"
    friction: float = 0.9
    settle_clearance: float = 0.003


@dataclass
class PlatformConfig:
    robot: RobotCfg
    feet: FeetCfg
    gains: GainsCfg
    sim: SimCfg
    poses: PosesCfg

    @classmethod
    def load(cls, path: str | Path) -> "PlatformConfig":
        raw = yaml.safe_load(Path(path).read_text())
        gains = GainsCfg(
            groups={
                k: GainGroup(**v) for k, v in raw["gains"]["groups"].items()
            },
            patterns=raw["gains"]["patterns"],
            default=GainGroup(**raw["gains"].get(
                "default", {"kp": 50, "kv": 4, "torque_limit": 30}
            )),
        )
        return cls(
            robot=RobotCfg(**raw["robot"]),
            feet=FeetCfg(**raw["feet"]),
            gains=gains,
            sim=SimCfg(**raw.get("sim", {})),
            poses=PosesCfg(**raw.get("poses", {})),
        )
