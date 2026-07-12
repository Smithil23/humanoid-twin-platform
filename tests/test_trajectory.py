"""Trajectory generator properties."""
from htp.trajectory import cosine_ramp, squat_targets


def test_ramp_is_zero_at_ends():
    assert cosine_ramp(0.0, 4.0, 0.5) == 0.0
    assert cosine_ramp(4.0, 4.0, 0.5) == 0.0


def test_ramp_peaks_mid_cycle():
    assert abs(cosine_ramp(2.0, 4.0, 0.5) - 0.5) < 1e-9


def test_squat_keeps_torso_level():
    tgt = squat_targets(4.0, period=4.0, depth=0.5, start=2.0)
    for side in ("left", "right"):
        knee = tgt[f"{side}_knee_joint"]          # base 0 -> knee = -s
        assert tgt[f"{side}_hip_pitch_joint"] == -knee / 2
        assert tgt[f"{side}_ankle_pitch_joint"] == -knee / 2


def test_squat_overlays_base_pose():
    base = {"left_knee_joint": -1.0472, "left_hip_pitch_joint": 0.5236,
            "left_ankle_pitch_joint": 0.5236}
    tgt = squat_targets(4.0, period=4.0, depth=0.5, start=2.0, base=base)
    assert tgt["left_knee_joint"] > base["left_knee_joint"]      # deeper bend
    assert tgt["left_hip_pitch_joint"] < base["left_hip_pitch_joint"]
