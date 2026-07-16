# `walking/` — LIPM/ZMP walking controller

A proper pattern-generation walking stack to replace the reactive stride+lean
controller in `scripts/walk_forward.py`. It is the feedforward layer; your
existing balance stack (capture-point ankle + hip strategy) plugs in as the
feedback layer.

## Data flow

```
GaitParams ──► plan_footsteps ──► zmp_reference ──┐
                                                  ▼
              measured CoM/ZMP ──► Stabilizer ──► PreviewController (x,y)
              (balance.py)          (feedback)     (feedforward CoM)
                                                  │
                    swing_foot_pose ──────────────┤
                    (swing leg arc)               ▼
                                              pelvis + foot poses
                                                  │
                                                  ▼
                                        leg_ik (both legs) ──► sim.py
```

Each control tick (`WalkController.step()`):
1. Read the ZMP reference for the current sample + preview window.
2. Ask the `Stabilizer` for a ZMP correction from measured CoM/ZMP error.
3. Step the preview controller (x and y) → feedforward CoM.
4. Determine support/swing phase; build swing-foot world pose.
5. Place the pelvis at the CoM, offset hips ±`hip_offset_y`.
6. Solve `leg_ik` for both legs; send joint targets.

The preview runs **receding-horizon**, so `replan()` can inject or shift
footsteps mid-walk (push recovery via `stepper.py`) and the CoM pattern
adapts within one preview horizon.

## Two seams you implement

**`RobotIO`** (against `sim.py`) — `read_com()`, `read_zmp()`,
`send_leg_targets(left_q, right_q)`. Joint order per leg must match
`leg_ik`: `[hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]`.

**`Stabilizer`** (against `balance.py`) — `zmp_correction(...) -> (2,)`.
Return zeros for pure feedforward; wire your capture-point / ankle / hip
logic here to shift the desired ZMP toward the measured CoM error. This is
the single integration point between the new pattern generator and your
proven balance stack.

## Tuning notes

- **`com_height` must be < full leg length** (`thigh + shank`) so the knee
  stays bent — straight-knee is a kinematic singularity. Defaults ship at
  `com_height = 0.72`, `thigh = shank = 0.40` (knee ≈ 20° bent when standing).
- **Match `LegParams` and the `leg_ik` joint order/axis signs to STAR1's
  URDF.** The solver is verified correct for the standard 6-DOF leg topology
  (FK round-trip to machine precision), but STAR1's link lengths, any hip-yaw
  offset, and axis directions are yours to set. If `balance.py` already
  exposes leg IK matched to STAR1, prefer it and use `leg_ik` as a cross-check.
- **Preview horizon** (`preview_horizon`, default 1.6 s ≈ 2 steps): longer =
  smoother anticipation, more latency to plan changes. Below ~1 step it
  degrades.
- **`R` (jerk penalty, default 1e-6) vs `Qe` (ZMP error, 1.0):** lower `R` →
  tighter ZMP tracking, more aggressive CoM. Raise `R` if the CoM looks jerky.

## Verified behaviour (self-tests in each module's `__main__`)

- `preview.py`: ZMP tracking with realistic double-support ramps → **2–3 mm
  RMS**; CoM correctly *leads* the ZMP.
- `leg_ik.py`: FK round-trip over 3000 random configs → **0 µm** position,
  **~1e-15** rotation error.
- `controller.py`: full pipeline, 10 steps, CoM advances as planned, feet
  lift to target clearance, joint targets move **≤0.5°/tick** (smooth).

Run any module directly (`python -m walking.controller`) to reproduce.

## Known limitations / next steps

- Feet are assumed flat on level ground (`ground_z = 0`). Uneven terrain
  needs per-footstep height + a foot-pitch/roll term in the swing target.
- Pattern generation is open-loop in the footstep plan; genuine capture-point
  *stepping* (choosing where to step from the divergent component of motion)
  should drive `replan()` from `balance.py` rather than a fixed `n_steps`.
- No arm/torso compensation for yaw momentum; add if turning-in-place induces
  spin.
