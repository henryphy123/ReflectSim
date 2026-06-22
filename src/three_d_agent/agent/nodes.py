"""LangGraph node implementations + reflect router."""
import json
from pathlib import Path
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from three_d_agent.agent.json_utils import parse_llm_json
from three_d_agent.agent.prompts import (
    PLANNER_PROMPT,
    SYSTEM_PROMPT,
    TUNE_SYSTEM_PROMPT,
    TUNE_USER_PROMPT,
)
from three_d_agent.agent.state import (
    AgentState,
    RobotResult,
    SceneObject,
    TaskGoal,
)
from three_d_agent.sim.config import SimConfig
from three_d_agent.sim.scene_builder import BuiltScene
from three_d_agent.tools.decompose_scene import decompose_scene
from three_d_agent.tools.fetch_assets import fetch_assets
from three_d_agent.tools.solve_pose import solve_pose
from three_d_agent.tools.validate_robot import validate_robot


_MAX_TUNE_ITERS = 3


_SCENE_CACHE: dict[str, BuiltScene] = {}


def node_planner(state: AgentState, llm: BaseChatModel) -> dict:
    msg = PLANNER_PROMPT.format(user_input=state.user_input)
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                           HumanMessage(content=msg)])
    text = response.content if isinstance(response.content, str) else str(response.content)
    try:
        plan = parse_llm_json(text)
    except ValueError:
        plan = ["decompose_scene", "fetch_assets", "solve_poses",
                "build_scene", "validate_robot"]
    return {"plan": plan, "messages_log": state.messages_log + ["planner"]}


def node_decompose(state: AgentState, llm: BaseChatModel) -> dict:
    result = decompose_scene(state.user_input, state.input_image_path, llm=llm)
    objects = [SceneObject(**o) for o in result["objects"]]
    task = TaskGoal(**result["task"])
    return {
        "objects": objects,
        "task": task,
        "messages_log": state.messages_log + ["decompose"],
    }


def node_fetch_assets(
    state: AgentState,
    retrieve: Callable,
    generate: Callable,
) -> dict:
    placed = fetch_assets(
        state.objects, retrieve=retrieve, generate=generate,
        image_path=state.input_image_path,
    )
    return {"objects": placed, "messages_log": state.messages_log + ["fetch_assets"]}


def node_solve_poses(state: AgentState, llm: BaseChatModel) -> dict:
    constraints = [f"{o.name} on table" for o in state.objects if not o.is_fixed]
    placed = solve_pose(
        state.objects,
        constraints=constraints,
        table_bbox=((0.0, -0.4, 0.20), (0.8, 0.4, 0.25)),
        llm=llm,
    )
    return {"objects": placed, "messages_log": state.messages_log + ["solve_poses"]}


def node_build_scene(state: AgentState) -> dict:
    from three_d_agent.sim.scene_builder import build_scene as _build
    scene_id = (
        f"scene_{abs(hash(state.user_input)) % 10**6}"
        f"_t{state.tune_iterations}"
    )
    try:
        built: BuiltScene = _build(
            state.objects, scene_id=scene_id, show_viewer=False,
            sim_config=state.sim_config,
        )
    except TypeError:
        built = _build(state.objects, scene_id=scene_id, show_viewer=False)
    _SCENE_CACHE[scene_id] = built
    return {
        "scene_id": scene_id,
        "scene_built": True,
        "messages_log": state.messages_log + ["build_scene"],
    }


def node_validate(state: AgentState, video_dir: Path) -> dict:
    if state.task is None or state.scene_id is None:
        return {"error": "missing task or scene_id",
                "messages_log": state.messages_log + ["validate(skipped)"]}
    built = _SCENE_CACHE[state.scene_id]
    result: RobotResult = validate_robot(
        built, state.task, state.objects, n_trials=3,
        video_dir=video_dir, sim_config=state.sim_config,
    )
    return {
        "robot_result": result,
        "video_path": result.video_path,
        "last_telemetries": result.telemetries,
        "messages_log": state.messages_log + ["validate"],
    }


def node_tune_config(state: AgentState, llm: BaseChatModel) -> dict:
    """Ask the LLM to propose an updated SimConfig given last-batch telemetry.

    Mirrors `scripts/llm_self_tune.py:ask_llm_for_new_config`. After producing
    a new config we also reset the per-iteration scene state so the next
    iteration rebuilds from scratch (otherwise the cached BuiltScene would
    silently use the old SimConfig).
    """
    schema = SimConfig.model_json_schema()
    n_trials = len(state.last_telemetries)
    n_success = sum(1 for t in state.last_telemetries if t.success)
    user_msg = TUNE_USER_PROMPT.format(
        config_schema=json.dumps(schema, indent=2),
        current_config=state.sim_config.model_dump_json(indent=2),
        n_trials=n_trials,
        n_success=n_success,
        telemetry=json.dumps(
            [t.model_dump() for t in state.last_telemetries], indent=2
        ),
        allow_attach_toggle="ALLOWED",
    )
    response = llm.invoke([
        SystemMessage(content=TUNE_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])
    text = (response.content if isinstance(response.content, str)
            else str(response.content))
    payload = parse_llm_json(text)
    new_cfg = SimConfig.model_validate(payload["config"])
    reasoning = str(payload.get("reasoning", ""))

    return {
        "sim_config": new_cfg,
        "tune_iterations": state.tune_iterations + 1,
        "scene_built": False,
        "scene_id": None,
        "robot_result": None,
        "messages_log": state.messages_log + [
            f"tune_config(iter={state.tune_iterations + 1}): "
            f"{reasoning[:160]}"
        ],
    }


def node_reflect(state: AgentState, llm: BaseChatModel) -> dict:
    return {
        "retry_count": state.retry_count + 1,
        "messages_log": state.messages_log + ["reflect"],
    }


def reflect_router(state: AgentState) -> str:
    if state.retry_count > 2 and state.tune_iterations >= _MAX_TUNE_ITERS:
        return "give_up"
    if state.robot_result is None:
        return "retry_assets"
    rate = state.robot_result.success_rate
    if rate >= 1 / 3:
        return "success"
    if state.tune_iterations < _MAX_TUNE_ITERS:
        return "retry_with_tuned_config"
    if rate == 0.0:
        return "retry_poses"
    return "success"
