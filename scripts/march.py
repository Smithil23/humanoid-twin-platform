#!/usr/bin/env python3
"""Sustained marching in place - the verified recipe (162 steps / 2 min).

Opens the MuJoCo viewer and marches indefinitely using the hip-strategy
balance controller (ankle + waist + arm counter-swing). This is the
reference implementation; the Twin Studio March button uses the same
logic.

    python scripts/march.py
"""
import time

import mujoco
import mujoco.viewer

from htp import PlatformConfig, Simulator
from htp.balance import BalanceController
from htp.stepper import Stepper

ARM_GAIN = 2.5


def main() -> None:
    sim = Simulator(PlatformConfig.load("configs/star1_visual.yaml"))
    bal = BalanceController(sim, hip_kp=2.5, hip_kd=0.5, hip_max=0.35)
    st = Stepper(sim)
    st.p.t_settle = 0.8
    st.p.amp = 0.08
    st.reset()
    base = dict(sim.cfg.poses.stand)
    dt = sim.cfg.sim.timestep
    # settle
    for _ in range(int(2.0 / dt)):
        mujoco.mj_step(sim.model, sim.data)

    with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
        while v.is_running():
            t0 = time.time()
            jt, ref = st.update(dt)
            op, orr = bal.update(dt, ref=ref)
            sides = bal.stance_sides()
            hp, hr = bal.hip_offsets()
            tgt = dict(base)
            tgt.update(jt)
            for j in BalanceController.ANKLE_PITCH:
                if j.split("_")[0] in sides:
                    tgt[j] = tgt.get(j, 0.0) + op
            for j in BalanceController.ANKLE_ROLL:
                if j.split("_")[0] in sides:
                    tgt[j] = tgt.get(j, 0.0) + orr
            tgt["waist_pitch_joint"] = tgt.get("waist_pitch_joint", 0.0) + hp
            tgt["waist_roll_joint"] = tgt.get("waist_roll_joint", 0.0) + hr
            for sh in ("left_shoulder_roll_joint",
                       "right_shoulder_roll_joint"):
                tgt[sh] = tgt.get(sh, 0.0) + ARM_GAIN * hr
            sim.set_joint_targets(tgt)
            sim.step()
            v.sync()
            left = dt - (time.time() - t0)
            if left > 0:
                time.sleep(left)


if __name__ == "__main__":
    main()
