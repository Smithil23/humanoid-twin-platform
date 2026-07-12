#!/usr/bin/env python3
"""Interactive 3D view of the running twin (MuJoCo viewer).

Shows the complete robot. Use configs/star1_visual.yaml to render the
real CAD meshes (requires the mesh files on disk); the default config
shows contact geometry only.

    python scripts/view_live.py --config configs/star1_visual.yaml --squat

Viewer controls: drag = orbit, right-drag = pan, scroll = zoom,
double-click a body to track it, Space pauses physics.
"""
import argparse
import time

import mujoco
import mujoco.viewer

from htp import PlatformConfig, Simulator
from htp.trajectory import squat_targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/star1_visual.yaml")
    ap.add_argument("--squat", action="store_true",
                    help="run squat cycles instead of standing still")
    args = ap.parse_args()

    sim = Simulator(PlatformConfig.load(args.config))
    dt = sim.cfg.sim.timestep

    with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
        while v.is_running():
            t0 = time.time()
            if args.squat:
                # continuous squat cycles, 6 s period
                sim.set_joint_targets(
                    squat_targets(sim.data.time % 6.0, period=4.0, depth=1.0,
                                  start=1.0, base=sim.cfg.poses.stand)
                )
            sim.step()
            v.sync()
            # soft real-time pacing
            leftover = dt - (time.time() - t0)
            if leftover > 0:
                time.sleep(leftover)


if __name__ == "__main__":
    main()
