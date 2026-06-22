"""Pydantic state models for the agent."""
from pathlib import Path
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, computed_field

from three_d_agent.sim.config import SimConfig, TrialTelemetry


class Pose(BaseModel):
    position: tuple[float, float, float]
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


class SceneObject(BaseModel):
    name: str
    description: str
    color: str | None = None
    size_hint: str | None = None
    shape_hint: Literal["box", "sphere", "cylinder", "mesh"] | None = None
    source: Literal["retrieved", "generated"] | None = None
    mesh_path: Path | None = None
    pose: Pose | None = None
    is_fixed: bool = False


class TaskGoal(BaseModel):
    action: Literal["pick_and_place", "stack", "push"]
    subject: str
    target: str | None = None


class RobotResult(BaseModel):
    n_trials: int
    n_success: int
    video_path: Path | None = None
    failure_modes: list[str] = Field(default_factory=list)
    telemetries: list[TrialTelemetry] = Field(default_factory=list)

    @computed_field
    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_trials if self.n_trials else 0.0


class AgentState(BaseModel):
    user_input: str
    input_image_path: Path | None = None
    plan: list[str] = Field(default_factory=list)
    objects: list[SceneObject] = Field(default_factory=list)
    task: TaskGoal | None = None
    scene_id: str | None = None
    scene_built: bool = False
    robot_result: RobotResult | None = None
    video_path: Path | None = None
    error: str | None = None
    retry_count: int = 0
    sim_config: SimConfig = Field(default_factory=SimConfig)
    last_telemetries: list[TrialTelemetry] = Field(default_factory=list)
    tune_iterations: int = 0
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    messages_log: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}
