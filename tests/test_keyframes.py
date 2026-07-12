"""Keyframe player: interpolation, looping, and physics stability."""
import pytest

from htp import PlatformConfig, Simulator
from htp.keyframes import KeyframePlayer, list_motions

SPEC = {
    "name": "test", "duration": 4.0, "loop": True,
    "keys": [
        {"t": 0.0, "pose": {"j": 0.0}},
        {"t": 2.0, "pose": {"j": 1.0}},
    ],
}


def test_endpoints_and_midpoint():
    p = KeyframePlayer(SPEC)
    assert p.targets(0.0)["j"] == pytest.approx(0.0)
    assert p.targets(2.0)["j"] == pytest.approx(1.0)
    assert p.targets(1.0)["j"] == pytest.approx(0.5)   # cosine mid = 0.5


def test_loop_wraps_smoothly():
    p = KeyframePlayer(SPEC)
    assert p.targets(4.0)["j"] == pytest.approx(p.targets(0.0)["j"])
    assert p.targets(3.0)["j"] == pytest.approx(0.5)   # halfway back down


def test_base_pose_overlay():
    p = KeyframePlayer(SPEC, base={"other": 0.7})
    tgt = p.targets(1.0)
    assert tgt["other"] == 0.7 and "j" in tgt


def test_motions_folder_lists_files():
    motions = list_motions()
    assert "wave" in motions and "bow" in motions


@pytest.mark.parametrize("motion", ["wave", "bow"])
def test_motion_keeps_robot_upright(motion):
    sim = Simulator(PlatformConfig.load("configs/star1.yaml"))
    player = KeyframePlayer.from_file(
        list_motions()[motion], base=sim.cfg.poses.stand)
    t0 = None
    steps = int(8.0 / sim.cfg.sim.timestep)
    for i in range(steps):
        t = sim.data.time if t0 is None else sim.data.time - t0
        sim.set_joint_targets(player.targets(t))
        sim.step()
    assert sim.upright, f"{motion} tipped the robot"
    assert sim.balance_margin() > 0
