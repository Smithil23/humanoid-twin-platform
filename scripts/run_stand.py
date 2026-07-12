#!/usr/bin/env python3
"""Standing test: hold home pose on two feet for --seconds."""
import argparse

from htp import PlatformConfig, RunLog, Simulator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/star1.yaml")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default=None, help="save log as .npz")
    args = ap.parse_args()

    sim = Simulator(PlatformConfig.load(args.config))
    log = RunLog(sim.hinge_names)
    steps = int(args.seconds / sim.cfg.sim.timestep)
    for i in range(steps):
        sim.step()
        if i % 25 == 0:
            log.add(sim.state())
    s = sim.state()
    print(f"t={s.time:.1f}s  base z={s.base_height:.3f} m  "
          f"contacts={s.n_contacts}  Fz={s.contact_force_z:.0f} N  "
          f"upright={sim.upright}")
    if args.out:
        log.save_npz(args.out)
        print("log saved to", args.out)


if __name__ == "__main__":
    main()
