"""LLM-in-the-loop self-tuning for the pick-and-place SimConfig.

Drops into the project's existing sim stack. Each iteration:
  1. Build the scene with the current SimConfig.
  2. Run N pick-and-place trials, collect TrialTelemetry.
  3. If success_rate >= target_success_rate, stop.
  4. Otherwise feed (current config + trial telemetry) to Claude and ask for a
     new SimConfig in JSON. Parse, apply, repeat until max_iter.

Usage (Azure-routed Claude):
  ANTHROPIC_API_KEY=<token> ANTHROPIC_BASE_URL=<url> \
  python scripts/llm_self_tune.py --max-iter 5 --target 1.0

Why this matters: shows the reflect loop tuning hyperparameters from
observation, not from human edits. Exactly what an automated robot training
data pipeline needs — the system that builds the scene also tunes the policy
that validates it.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from three_d_agent.agent.json_utils import parse_llm_json
from three_d_agent.agent.llm import LLMConfig, build_llm
from three_d_agent.agent.prompts import TUNE_SYSTEM_PROMPT, TUNE_USER_PROMPT
from three_d_agent.agent.state import Pose, SceneObject, TaskGoal
from three_d_agent.sim.config import SimConfig, TrialTelemetry, TuneAttempt
from three_d_agent.sim.robot import run_pick_and_place
from three_d_agent.sim.scene_builder import build_scene


SYSTEM_PROMPT = TUNE_SYSTEM_PROMPT
TUNE_PROMPT = TUNE_USER_PROMPT


@dataclass
class TuneArgs:
    max_iter: int
    target_success_rate: float
    n_trials_per_config: int
    allow_attach_toggle: bool
    log_path: Path
    model: str


def _print_header(args: TuneArgs) -> None:
    print("=" * 72)
    print("LLM self-tuning loop")
    print(f"  max_iter           : {args.max_iter}")
    print(f"  target_success_rate: {args.target_success_rate}")
    print(f"  trials per config  : {args.n_trials_per_config}")
    print(f"  allow attach toggle: {args.allow_attach_toggle}")
    print(f"  llm model          : {args.model}")
    print(f"  log file           : {args.log_path}")
    print("=" * 72)


def run_one_config(cfg: SimConfig, n_trials: int) -> tuple[int, list[TrialTelemetry]]:
    """Run n_trials pick-and-place attempts with this config; return success
    count and per-trial telemetry. Builds a fresh scene per trial so failures
    don't carry over.
    """
    n_success = 0
    telemetries: list[TrialTelemetry] = []
    for i in range(n_trials):
        objs = [
            SceneObject(name="cube", description="small red cube",
                        color="red", size_hint="small",
                        pose=Pose(position=(0.40, 0.0, 0.30))),
        ]
        built = build_scene(objs, scene_id=f"tune_trial_{i}", sim_config=cfg)
        task = TaskGoal(action="pick_and_place", subject="cube", target=None)
        _, telemetry = run_pick_and_place(built, task, sim_config=cfg)
        if telemetry.success:
            n_success += 1
        telemetries.append(telemetry)
    return n_success, telemetries


def ask_llm_for_new_config(
    llm: Any,
    current: SimConfig,
    telemetries: list[TrialTelemetry],
    n_success: int,
    *,
    allow_attach_toggle: bool,
) -> tuple[SimConfig, str]:
    """Ask the LLM to propose an updated SimConfig given telemetry."""
    schema = SimConfig.model_json_schema()
    msg = TUNE_USER_PROMPT.format(
        config_schema=json.dumps(schema, indent=2),
        current_config=current.model_dump_json(indent=2),
        n_trials=len(telemetries),
        n_success=n_success,
        telemetry=json.dumps(
            [t.model_dump() for t in telemetries], indent=2
        ),
        allow_attach_toggle="ALLOWED" if allow_attach_toggle else "NOT ALLOWED",
    )
    from langchain_core.messages import HumanMessage, SystemMessage
    response = llm.invoke([
        SystemMessage(content=TUNE_SYSTEM_PROMPT),
        HumanMessage(content=msg),
    ])
    text = (response.content if isinstance(response.content, str)
            else str(response.content))
    payload = parse_llm_json(text)
    reasoning = str(payload.get("reasoning", "(no reasoning provided)"))
    new_cfg_dict = payload["config"]
    if not allow_attach_toggle:
        new_cfg_dict["use_attach_on_close"] = current.use_attach_on_close
    new_cfg = SimConfig.model_validate(new_cfg_dict)
    return new_cfg, reasoning


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--target", type=float, default=1.0,
                        help="Stop when success rate >= this (default 1.0).")
    parser.add_argument("--n-trials", type=int, default=3,
                        help="Trials per SimConfig (default 3).")
    parser.add_argument("--allow-attach-toggle", action="store_true",
                        help="Let the LLM flip use_attach_on_close. Default OFF "
                             "(forces it to find a friction-based grasp).")
    parser.add_argument("--start-attach-off", action="store_true",
                        help="Start with use_attach_on_close=False (the harder "
                             "starting point — friction grasp from scratch).")
    parser.add_argument("--model", default="gsds-claude-sonnet-4-5")
    parser.add_argument("--log", default="runs/tune_log.json")
    args_ns = parser.parse_args(argv)
    args = TuneArgs(
        max_iter=args_ns.max_iter,
        target_success_rate=args_ns.target,
        n_trials_per_config=args_ns.n_trials,
        allow_attach_toggle=args_ns.allow_attach_toggle,
        log_path=Path(args_ns.log),
        model=args_ns.model,
    )
    _print_header(args)

    cfg = SimConfig(use_attach_on_close=not args_ns.start_attach_off)
    llm = build_llm(LLMConfig(provider="anthropic", model=args.model,
                              max_tokens=4096))
    history: list[TuneAttempt] = []

    for it in range(args.max_iter):
        print(f"\n--- iteration {it + 1}/{args.max_iter} ---")
        print(f"config: {cfg.model_dump_json()}")
        n_success, tels = run_one_config(cfg, args.n_trials_per_config)
        rate = n_success / args.n_trials_per_config
        print(f"  result: {n_success}/{args.n_trials_per_config} = {rate:.2f}")
        for t in tels:
            print(f"  - {t.phase_at_failure:<8s} "
                  f"lift={t.cube_vertical_lift_m * 100:+5.1f} cm "
                  f"horiz={t.cube_horizontal_displacement_m * 100:+5.1f} cm "
                  f"finger={t.final_finger_qpos:.3f}")

        reasoning = ""
        if rate >= args.target_success_rate:
            print(f"  HIT target ({rate} >= {args.target_success_rate}). "
                  f"Stopping.")
            history.append(TuneAttempt(iteration=it, config=cfg,
                                        telemetry=tels, success_rate=rate,
                                        llm_reasoning=""))
            break

        print("  asking LLM for an improved config ...")
        cfg, reasoning = ask_llm_for_new_config(
            llm, cfg, tels, n_success,
            allow_attach_toggle=args.allow_attach_toggle,
        )
        print(f"  LLM reasoning: {reasoning}")
        history.append(TuneAttempt(iteration=it, config=cfg,
                                    telemetry=tels, success_rate=rate,
                                    llm_reasoning=reasoning))

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text(
        json.dumps([h.model_dump() for h in history], indent=2)
    )
    print(f"\nwrote log to {args.log_path}")
    print("=" * 72)
    if history and history[-1].success_rate >= args.target_success_rate:
        print(f"SUCCESS: final config achieves "
              f"{history[-1].success_rate:.2f} success rate after "
              f"{history[-1].iteration + 1} iterations.")
        return 0
    print(f"FAILED to hit target {args.target_success_rate} in "
          f"{args.max_iter} iterations.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
