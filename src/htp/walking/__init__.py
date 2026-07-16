"""LIPM/ZMP walking controller for the STAR1 humanoid twin platform.

Modules:
    preview     — ZMP-preview CoM trajectory (Kajita 2003 cart-table)
    footsteps   — footstep planner + ZMP reference
    swing       — swing-foot trajectory
    leg_ik      — analytic 6-DOF leg inverse kinematics
    controller  — online orchestrator (RobotIO / Stabilizer seams)
"""
from .preview import PreviewController, generate_com_trajectory
from .footsteps import GaitParams, Side, Footstep, plan_footsteps, zmp_reference
from .swing import swing_foot_pose
from .leg_ik import LegParams, leg_ik, leg_fk
from .controller import WalkController, WalkConfig, RobotIO, Stabilizer
