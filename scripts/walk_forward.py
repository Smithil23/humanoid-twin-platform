#!/usr/bin/env python3
"""Forward walking - WORK IN PROGRESS.

Extends the sustained in-place march (scripts/march.py) with a forward
stride and a phase-synced forward lean. Current status: the robot walks
forward ~15-20 cm under control, then accumulated forward momentum
outpaces the ankle+hip balance recovery and it falls (~15-20 s).

This is the expected difficulty wall: robust forward walking needs
planned footstep placement with a CoM trajectory that arrives over each
new foot (LIPM / ZMP preview control), not a stride offset bolted onto
the in-place stepper. The march (scripts/march.py) is stable and
indefinite; this script is the forward-walking research frontier.

    python scripts/walk_forward.py

Tuning knobs at the top: STRIDE (forward hip swing), LEAN (double-support
forward CoM push). Larger values travel faster but fall sooner.
"""
import time

import mujoco
import mujoco.viewer

from htp import PlatformConfig, Simulator
from htp.balance import BalanceController
from htp.stepper import Stepper

STRIDE = 0.08     # forward hip-pitch swing [rad]
LEAN = 0.04       # forward CoM lean during double support [m]
ARM_GAIN = 2.5


def main() -> None:
    sim = Simulator(PlatformConfig.load("configs/star1_visual.yaml"))
    bal = BalanceController(sim, hip_kp=2.5, hip_kd=0.5, hip_max=0.35)
    st = Stepper(sim)
    st.p.t_settle = 0.8
    st.p.amp = 0.08
    st.p.stride = STRIDE
    st.reset()
    base = dict(sim.cfg.poses.stand)
    dt = sim.cfg.sim.timestep
    for _ in range(int(2.0 / dt)):
        mujoco.mj_step(sim.model, sim.data)
    x0 = float(sim.data.subtree_com[1][0])

    with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
        while v.is_running():
            t0 = time.time()
            jt, ref = st.update(dt)
            lean = LEAN if st.phase in ("SETTLE", "RETURN") else 0.0
            op, orr = bal.update(dt, ref=(lean, ref[1]))
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
            if not sim.upright:
                dist = (float(sim.data.subtree_com[1][0]) - x0) * 100
                print(f"fell after walking {dist:+.0f} cm forward")
                break
            left = dt - (time.time() - t0)
            if left > 0:
                time.sleep(left)


if __name__ == "__main__":
    main()
