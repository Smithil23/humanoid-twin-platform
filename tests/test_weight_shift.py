"""Weight-shift motion: full cycles transfer load foot-to-foot while
the balance controller keeps the robot stable."""
from htp import PlatformConfig, Simulator
from htp.balance import BalanceController
from htp.keyframes import KeyframePlayer, list_motions


def test_weight_shift_transfers_load_and_stays_up():
    sim = Simulator(PlatformConfig.load("configs/star1.yaml"))
    bal = BalanceController(sim)
    player = KeyframePlayer.from_file(
        list_motions()["weight_shift"], base=sim.cfg.poses.stand)
    dt = sim.cfg.sim.timestep
    sim.step(int(2.0 / dt))
    t0 = sim.data.time
    lmin = rmin = 1e9
    lmax = rmax = 0.0
    while sim.data.time < t0 + 12.5:      # two full cycles
        tgt = player.targets(sim.data.time - t0)
        ref = (tgt.pop("com_x", 0.0), tgt.pop("com_y", 0.0))
        op, orr = bal.update(dt, ref=ref)
        hp, hr = bal.hip_offsets()      # hip strategy is part of the
        for j in BalanceController.ANKLE_PITCH:   # controller now; apply
            tgt[j] = tgt.get(j, 0.0) + op         # its full output as
        for j in BalanceController.ANKLE_ROLL:    # the Studio does
            tgt[j] = tgt.get(j, 0.0) + orr
        wp, wr = BalanceController.WAIST
        tgt[wp] = tgt.get(wp, 0.0) + hp
        tgt[wr] = tgt.get(wr, 0.0) + hr
        sim.set_joint_targets(tgt)
        sim.step()
        assert sim.upright, "fell during weight shift"
        if int(sim.data.time / dt) % 100 == 0:
            ff = sim.foot_forces()
            left, right = (ff["left_ankle_roll_link"],
                           ff["right_ankle_roll_link"])
            lmin, rmin = min(lmin, left), min(rmin, right)
            lmax, rmax = max(lmax, left), max(rmax, right)
    # meaningful load transfer: each foot swings by > 270 N over a cycle
    assert lmin < 150 and rmin < 230, "feet never unloaded"
    assert lmax > 420 and rmax > 420, "feet never loaded"
    assert sim.balance_margin() > 0
