
from functools import partial
from pathlib import Path
from typing import Callable, Literal

from langgraph.graph import END, START, StateGraph

from three_d_agent.agent.llm import LLMConfig, build_llm
from three_d_agent.agent.state import AgentState

ReflectDecision = Literal[
    "success", "retry_poses", "retry_assets",
    "retry_with_tuned_config", "give_up",
]


def _stub_planner(state):       return {"plan": ["stub_planner_ran"], "messages_log": ["planner"]}
def _stub_decompose(state):     return {"messages_log": state.messages_log + ["decompose"]}
def _stub_fetch_assets(state):  return {"messages_log": state.messages_log + ["fetch_assets"]}
def _stub_solve_poses(state):   return {"messages_log": state.messages_log + ["solve_poses"]}
def _stub_build_scene(state):   return {"scene_built": True,
                                        "messages_log": state.messages_log + ["build_scene"]}
def _stub_validate(state):      return {"messages_log": state.messages_log + ["validate"]}
def _stub_reflect(state):       return {"messages_log": state.messages_log + ["reflect"]}
def _stub_tune_config(state):   return {"messages_log": state.messages_log + ["tune_config"]}
def _stub_reflect_router(state) -> ReflectDecision: return "success"


def build_graph(
    *,
    stub: bool = False,
    llm_config: LLMConfig | None = None,
    retrieve_fn: Callable | None = None,
    generate_fn: Callable | None = None,
    video_dir: Path | None = None,
    use_dummy_assets: bool = False,
):
    g = StateGraph(AgentState)

    if stub:
        g.add_node("planner",      _stub_planner)
        g.add_node("decompose",    _stub_decompose)
        g.add_node("fetch_assets", _stub_fetch_assets)
        g.add_node("solve_poses",  _stub_solve_poses)
        g.add_node("build_scene",  _stub_build_scene)
        g.add_node("validate",     _stub_validate)
        g.add_node("reflect",      _stub_reflect)
        g.add_node("tune_config",  _stub_tune_config)
        router = _stub_reflect_router
    else:
        from three_d_agent.agent import nodes
        llm = build_llm(llm_config or LLMConfig())
        if use_dummy_assets:
            from three_d_agent.tools.dummy_assets import dummy_retrieve, dummy_generate
            if retrieve_fn is None:
                retrieve_fn = dummy_retrieve
            if generate_fn is None:
                generate_fn = dummy_generate
        else:
            if retrieve_fn is None:
                from three_d_agent.tools.retrieve_asset import retrieve_asset
                retrieve_fn = retrieve_asset
            if generate_fn is None:
                from three_d_agent.tools.generate_asset import generate_asset
                generate_fn = generate_asset
        video_dir = video_dir or Path("runs")

        g.add_node("planner",      partial(nodes.node_planner, llm=llm))
        g.add_node("decompose",    partial(nodes.node_decompose, llm=llm))
        g.add_node("fetch_assets", partial(nodes.node_fetch_assets,
                                           retrieve=retrieve_fn,
                                           generate=generate_fn))
        g.add_node("solve_poses",  partial(nodes.node_solve_poses, llm=llm))
        g.add_node("build_scene",  nodes.node_build_scene)
        g.add_node("validate",     partial(nodes.node_validate, video_dir=video_dir))
        g.add_node("reflect",      partial(nodes.node_reflect, llm=llm))
        g.add_node("tune_config",  partial(nodes.node_tune_config, llm=llm))
        router = nodes.reflect_router

    g.add_edge(START, "planner")
    g.add_edge("planner", "decompose")
    g.add_edge("decompose", "fetch_assets")
    g.add_edge("fetch_assets", "solve_poses")
    g.add_edge("solve_poses", "build_scene")
    g.add_edge("build_scene", "validate")
    g.add_edge("validate", "reflect")
    g.add_edge("tune_config", "build_scene")
    g.add_conditional_edges("reflect", router, {
        "success":                 END,
        "retry_poses":             "solve_poses",
        "retry_assets":            "fetch_assets",
        "retry_with_tuned_config": "tune_config",
        "give_up":                 END,
    })
    return g.compile()
