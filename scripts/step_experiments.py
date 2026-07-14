#!/usr/bin/env python3
"""Stepping-in-place experiments - status and best-known parameters.

FINDINGS (session log):
- Open-loop keyframed stepping achieves 4 consecutive steps (2 full
  cycles, ~9.7 s) with the parameters below, then falls from
  accumulated drift: fixed timestamps lift the swing foot whether or
  not weight transfer actually completed, and landing errors compound.
- Single-leg HOLD is bounded at ~0.7 s by lateral ankle authority
  (measured); brief step-swing phases (~0.5 s) are survivable, which
  is why stepping nearly works while holding cannot.
- Conclusion: sustained stepping needs EVENT-BASED phase transitions
  (a step state machine), not better timing constants:
      SHIFT   -> until swing-foot force < threshold for N ms
      LIFT    -> swing knee/hip curl, fixed short duration
      LAND    -> extend until swing-foot force > threshold
      MIRROR  -> repeat opposite side
  plus per-step re-anchoring of the CoM reference to the new stance
  foot. That module (htp/stepper.py) is the next milestone.

Best known open-loop parameters (4 steps):
    amp=0.07 waist=0.30 hip_d=0.20 knee_d=0.40 half=2.4 lift=0.5

Run:  python scripts/step_experiments.py
"""
import mujoco  # noqa: F401

from htp import PlatformConfig, Simulator
from htp.balance import BalanceController
from htp.keyframes import KeyframePlayer

S, K = 0.5236, -1.0472
BEST = dict(amp=0.07, waist=0.30, hip_d=0.20, knee_d=0.40,
            half=2.4, lift=0.5)


def step_spec(amp, waist, hip_d, knee_d, half, lift):
    s1, l1 = half * 0.5, half * 0.625
    keys = [{"t": 0.0, "pose": {"com_y": 0.0, "waist_roll_joint": 0.0}}]
    for k, (sgn, leg) in enumerate(((-1, "left"), (+1, "right"))):
        o = k * half
        keys += [
            {"t": o + s1, "pose": {"com_y": sgn * amp,
                                   "waist_roll_joint": sgn * waist}},
            {"t": o + l1, "pose": {f"{leg}_hip_pitch_joint": S,
                                   f"{leg}_knee_joint": K}},
            {"t": o + l1 + lift * 0.5,
             "pose": {f"{leg}_hip_pitch_joint": S + hip_d,
                      f"{leg}_knee_joint": K + knee_d}},
            {"t": o + l1 + lift, "pose": {f"{leg}_hip_pitch_joint": S,
                                          f"{leg}_knee_joint": K}},
            {"t": o + half, "pose": {"com_y": 0.0,
                                     "waist_roll_joint": 0.0}},
        ]
    return {"name": "step", "duration": 2 * half, "loop": True,
            "keys": keys}


def run(cycles: int = 3, **p) -> None:
    sim = Simulator(PlatformConfig.load("configs/star1.yaml"))
    bal = BalanceController(sim)
    player = KeyframePlayer(step_spec(**p), base=sim.cfg.poses.stand)
    dt = sim.cfg.sim.timestep
    sim.step(int(2.0 / dt))
    t0 = sim.data.time
    air = {"left": 0.0, "right": 0.0}
    steps = 0
    was_air = {"left": False, "right": False}
    while sim.data.time < t0 + cycles * 2 * p["half"]:
        tgt = player.targets(sim.data.time - t0)
        ref = (tgt.pop("com_x", 0.0), tgt.pop("com_y", 0.0))
        op, orr = bal.update(dt, ref=ref)
        sides = bal.stance_sides()
        for j in BalanceController.ANKLE_PITCH:
            if j.split("_")[0] in sides:
                tgt[j] = tgt.get(j, 0.0) + op
        for j in BalanceController.ANKLE_ROLL:
            if j.split("_")[0] in sides:
                tgt[j] = tgt.get(j, 0.0) + orr
        sim.set_joint_targets(tgt)
        sim.step()
        if not sim.upright:
            print(f"fell at t={sim.data.time - t0:.1f}s "
                  f"after {steps} steps")
            return
        ff = sim.foot_forces()
        for side, link in (("left", "left_ankle_roll_link"),
                           ("right", "right_ankle_roll_link")):
            a = ff[link] < 5
            if a:
                air[side] += dt
            if a and not was_air[side]:
                steps += 1
            was_air[side] = a
    print(f"completed {cycles} cycles: {steps} steps, "
          f"air L {air['left']:.2f}s R {air['right']:.2f}s")


if __name__ == "__main__":
    run(**BEST)
