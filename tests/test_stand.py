"""Physics regression: the robot must stand for 5 simulated seconds."""
import pytest

from htp import PlatformConfig, Simulator


@pytest.fixture(scope="module")
def sim():
    return Simulator(PlatformConfig.load("configs/star1.yaml"))


def test_stands_for_five_seconds(sim):
    sim.reset()
    h0 = sim.state().base_height
    sim.step(int(5.0 / sim.cfg.sim.timestep))
    s = sim.state()
    assert sim.upright, "robot tipped over"
    assert abs(s.base_height - h0) < 0.05, "base height drifted"
    assert s.n_contacts >= 4, "feet lost ground contact"
    assert s.contact_force_z == pytest.approx(64 * 9.81, rel=0.3)
