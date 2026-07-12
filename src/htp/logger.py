"""Run logging: in-memory buffer -> npz / csv / mat exports."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .sim import SimState


class RunLog:
    """Collects SimState samples and exports them for analysis."""

    def __init__(self, joint_names: list[str]):
        self.joint_names = joint_names
        self._rows: list[SimState] = []

    def add(self, s: SimState) -> None:
        self._rows.append(s)

    def __len__(self) -> int:
        return len(self._rows)

    # ------------------------------------------------------------ exports
    def arrays(self) -> dict[str, np.ndarray]:
        r = self._rows
        return {
            "time": np.array([s.time for s in r]),
            "base_height": np.array([s.base_height for s in r]),
            "com": np.stack([s.com for s in r]),
            "joint_pos": np.stack([s.joint_pos for s in r]),
            "actuator_torque": np.stack([s.actuator_torque for s in r]),
            "contact_force_z": np.array([s.contact_force_z for s in r]),
            "qpos": np.stack([s.qpos for s in r]),
            "qvel": np.stack([s.qvel for s in r]),
        }

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            path, joint_names=np.array(self.joint_names), **self.arrays()
        )

    def save_mat(self, path: str | Path) -> None:
        """Export for the MATLAB validation step (see matlab/)."""
        from scipy.io import savemat

        data = self.arrays()
        data["joint_names"] = np.array(self.joint_names, dtype=object)
        savemat(str(path), data)
