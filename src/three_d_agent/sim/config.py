"""Tunable sim/control hyperparameters surfaced to the LLM self-tuning loop.

`SimConfig` is the single source of truth for every magic number that used to
live as a module-level constant in `scene_builder.py` or `robot.py`. The
self-tuning loop (`scripts/llm_self_tune.py`) treats it as the search space.

`TrialTelemetry` is what `run_pick_and_place` reports back to the loop so the
LLM can reason about *why* a trial failed, not just that it did.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SimConfig(BaseModel):
    """Hyperparameters the LLM is allowed to change between trials."""

    cube_half: float = Field(
        0.012, ge=0.005, le=0.05,
        description="Half-extent of the cube in metres. The Franka gripper "
                    "max-opens at 4 cm per side; cubes wider than 6-7 cm are "
                    "marginal even with a perfect grasp.")
    cube_mass: float = Field(
        0.02, ge=0.005, le=0.5,
        description="Cube mass in kg. Heavier cubes need stronger gripper "
                    "closing force; Franka's default tendon torque is weak.")
    cube_friction_slide: float = Field(
        2.0, ge=0.1, le=10.0,
        description="Sliding friction coefficient. Higher = more grip "
                    "purchase but also more knock-off resistance during "
                    "descent.")

    gripper_pregrasp_ctrl: float = Field(
        80.0, ge=0.0, le=255.0,
        description="Gripper ctrl value during the descent phase. 0 = fully "
                    "closed, 255 = fully open (~4 cm per side). Should be "
                    "set so finger width is just larger than cube_half so the "
                    "descent does not sweep the cube sideways.")

    tcp_offset_z: float = Field(
        0.105, ge=0.05, le=0.15,
        description="Vertical offset (m) between the Franka `hand` body frame "
                    "and the tool centre point (fingertip). Larger values "
                    "place fingertips lower for the same hand body target.")

    n_steps_per_trial: int = Field(
        2000, ge=500, le=8000,
        description="Total MuJoCo physics steps allocated across the five "
                    "waypoints in one pick-and-place trial. Too few = PD "
                    "controllers cannot track IK targets; too many = slow.")

    use_attach_on_close: bool = Field(
        True,
        description="If True, programmatically weld the cube to the hand once "
                    "the close-gripper waypoint completes (the standard "
                    "PyBullet/Isaac demo shortcut). If False, the lift "
                    "depends entirely on Franka's friction-based grip — much "
                    "harder to make work, but more physically honest.")

    wp_above_frac: float = Field(0.15, gt=0.0, le=0.5)
    wp_prenarrow_frac: float = Field(0.10, gt=0.0, le=0.5)
    wp_descend_frac: float = Field(0.20, gt=0.0, le=0.5)
    wp_close_frac: float = Field(0.15, gt=0.0, le=0.5)
    wp_lift_frac: float = Field(0.40, gt=0.0, le=0.7)

    @field_validator("wp_lift_frac")
    @classmethod
    def _fractions_sum_to_one(cls, v: float, info) -> float:
        return v

    model_config = {"json_schema_extra": {"x-llm-tunable": True}}


class TrialTelemetry(BaseModel):
    """What we tell the LLM about a single pick-and-place trial."""

    success: bool
    success_threshold_m: float = 0.05

    cube_start_xyz: tuple[float, float, float]
    cube_end_xyz: tuple[float, float, float]
    cube_horizontal_displacement_m: float
    cube_vertical_lift_m: float

    final_finger_qpos: float
    final_hand_xyz: tuple[float, float, float]
    final_hand_to_cube_distance_m: float

    phase_at_failure: Literal[
        "none",
        "descend",
        "close",
        "lift",
        "unknown",
    ]

    notes: list[str] = Field(default_factory=list)


class TuneAttempt(BaseModel):
    """One iteration of the LLM self-tuning loop."""

    iteration: int
    config: SimConfig
    telemetry: list[TrialTelemetry]
    success_rate: float
    llm_reasoning: str = ""
