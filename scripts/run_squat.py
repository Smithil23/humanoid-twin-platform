#!/usr/bin/env python3
"""Squat test: dynamic knee-bend cycle, exports a log for MATLAB validation."""
import argparse

from htp import PlatformConfig, RunLog, Simulator
from htp.trajectory import squat_targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/star1.yaml")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--depth", type=float, default=1.0)
    ap.add_argument("--mat", default=None, help="export .mat for MATLAB")
    ap.add_argument("--npz", default=None)
    args = ap.parse_args()

    sim = Simulator(PlatformConfig.load(args.config))
    log = RunLog(sim.hinge_names)
    steps = int(args.seconds / sim.cfg.sim.timestep)
    zmin = 10.0
    for i in range(steps):
        sim.set_joint_targets(squat_targets(sim.data.time, depth=args.depth, base=sim.cfg.poses.stand))
        sim.step()
        zmin = min(zmin, float(sim.data.qpos[2]))
        if i % 25 == 0:
            log.add(sim.state())
    s = sim.state()
    print(f"t={s.time:.1f}s  z end={s.base_height:.3f}  z min={zmin:.3f}  "
          f"upright={sim.upright}")
    if args.mat:
        log.save_mat(args.mat)
        print("MATLAB export:", args.mat)
    if args.npz:
        log.save_npz(args.npz)


if __name__ == "__main__":
    main()
