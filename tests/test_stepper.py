"""Stepper: event-driven phases fire on measured foot forces, the robot
takes real steps, and it does not fall within the tuned regime."""
from htp import PlatformConfig, Simulator
from htp.balance import BalanceController
from htp.stepper import Stepper


def _march(seconds: float):
    sim = Simulator(PlatformConfig.load("configs/star1.yaml"))
    bal = BalanceController(sim)
    st = Stepper(sim)
    st.reset()
    base = dict(sim.cfg.poses.stand)
    dt = sim.cfg.sim.timestep
    sim.step(int(2.0 / dt))
    t0 = sim.data.time
    phases = set()
    while sim.data.time < t0 + seconds:
        jt, ref = st.update(dt)
        op, orr = bal.update(dt, ref=ref)
        sides = bal.stance_sides()
        tgt = dict(base)
        tgt.update(jt)
        for j in BalanceController.ANKLE_PITCH:
            if j.split("_")[0] in sides:
                tgt[j] = tgt.get(j, 0.0) + op
        for j in BalanceController.ANKLE_ROLL:
            if j.split("_")[0] in sides:
                tgt[j] = tgt.get(j, 0.0) + orr
        sim.set_joint_targets(tgt)
        sim.step()
        phases.add(st.phase)
        if not sim.upright:
            return st.steps, phases, False
    return st.steps, phases, sim.upright


def test_stepper_cycles_through_phases_and_steps():
    steps, phases, upright = _march(20.0)
    # the state machine must actually run its cycle, not stall
    assert "LIFT" in phases and "LAND" in phases
    assert steps >= 2, f"expected real steps, got {steps}"
    assert upright, "robot fell within the tuned stepping regime"
