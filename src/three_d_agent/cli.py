"""CLI: three-d-agent "put a cup on a book" [--image scene.png]."""
import json
from pathlib import Path

import click
from dotenv import load_dotenv

from three_d_agent.agent.graph import build_graph
from three_d_agent.agent.state import AgentState
from three_d_agent.sim.config import SimConfig


@click.command()
@click.argument("user_input")
@click.option("--image", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Optional image input.")
@click.option("--video-dir", type=click.Path(),
              default="runs", show_default=True)
@click.option("--provider", default="anthropic", show_default=True)
@click.option("--model", default=None,
              help="Override LLM model id (else uses provider default).")
@click.option("--no-external", "use_dummy", is_flag=True, default=False,
              help="Use placeholder retrieve/generate. Lets you run a demo "
                   "without Objaverse/Hunyuan3D set up.")
@click.option("--start-sim-config", "start_sim_config", default=None,
              help="JSON dict of SimConfig overrides for the INITIAL config "
                   "(e.g. '{\"use_attach_on_close\": false}' to force the "
                   "tune loop to find a friction-based grasp).")
def main(user_input: str, image: str | None, video_dir: str,
         provider: str, model: str | None, use_dummy: bool,
         start_sim_config: str | None) -> None:
    load_dotenv()
    from three_d_agent.agent.llm import LLMConfig
    cfg_kwargs = {"provider": provider}
    if model:
        cfg_kwargs["model"] = model
    graph = build_graph(
        llm_config=LLMConfig(**cfg_kwargs),
        video_dir=Path(video_dir),
        use_dummy_assets=use_dummy,
    )
    sim_cfg = SimConfig()
    if start_sim_config:
        overrides = json.loads(start_sim_config)
        sim_cfg = SimConfig.model_validate(
            {**sim_cfg.model_dump(), **overrides}
        )
    state = AgentState(
        user_input=user_input,
        input_image_path=Path(image) if image else None,
        sim_config=sim_cfg,
    )
    final = graph.invoke(state)

    summary = {
        "scene_id": final.get("scene_id"),
        "video_path": str(final.get("video_path") or ""),
    }
    if final.get("robot_result"):
        r = final["robot_result"]
        summary.update({
            "success_rate": getattr(r, "success_rate", None),
            "n_success": getattr(r, "n_success", None),
            "n_trials": getattr(r, "n_trials", None),
        })
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
