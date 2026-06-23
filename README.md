# 3D Scene Generation Agent

> Natural language → physics-ready 3D scene → robot pick-and-place.
> A LangGraph agent for **automating robot training data generation**.

Industrial robot training is bottlenecked on hand-built 3D scenes — weeks per
scenario. This project collapses that into a one-shot pipeline: the agent
decomposes a sentence (or single image) into objects + a task, fetches or
generates the 3D assets, places them with collision-aware spatial reasoning,
assembles a MuJoCo scene with a Franka Panda arm, and validates by executing a
4-waypoint pick-and-place — under a minute, no manual modelling.

## Demo

![demo](docs/demo.gif)

Eight seconds of an end-to-end `--no-external` run; 3/3 success across the
trial loop. Reproduce it after install with:

```bash
three-d-agent --no-external "put a small red cube on a wooden table"
```

## Highlights

- **Agent orchestration** — LangGraph 7-node state machine with explicit
  `plan → act → reflect`, retry budget, per-tool dependency injection.
- **3D pipeline judgment** — retrieval-first (Objaverse-XL) / generation-fallback
  (Hunyuan3D-2); both bypassable via `--no-external` for offline CPU runs.
- **Spatial reasoning** — two-stage pose solver: LLM proposes coordinates, AABB
  collision filter rejects, LLM re-proposes. Not single-shot guessing.
- **Robotics literacy** — Franka Panda + damped-least-squares IK in MuJoCo,
  recorded via the offscreen renderer to MP4. Tracks waypoints to ~1 cm.
- ⭐ **Self-tuning closed loop** — `TrialTelemetry` → Claude → new `SimConfig`,
  rebuild, re-validate. Recorded run: **0/3 → 3/3 in one iteration**. See below.

## Self-tuning closed loop

The hard part of a robot demo isn't the LLM — it's the contact dynamics: cube
size, pre-grasp width, friction, settling time. The first working version of
this project hard-coded these. The next step is to let the agent tune itself.

`SimConfig` (`src/three_d_agent/sim/config.py`) exposes every hyperparameter
the grasp depends on. `TrialTelemetry` is what `run_pick_and_place` reports
back: knock-off distance, vertical lift, final finger position, phase-at-failure.
Together they let Claude reason about *why* a trial failed.

Recorded run starting with `use_attach_on_close=False` (forcing the harder
friction-based grasp; full log: `docs/tune_log_example.json`):

- **Iter 1** — 0/3 success. Telemetry: lift 0 cm, cube knocked sideways
  1.4 cm during descent, gripper closed but hand ended 29 cm from cube.
- **Claude's diagnosis** — *"gripper closed fully but the friction-based grip
  failed completely. Cube knocked sideways 1.4 cm during descent. I'll
  increase pregrasp width, raise friction, reduce cube mass, give close+lift
  more time."*
- **Iter 2** — **3/3 success**, +6.7 cm lift per trial, attach flag still off.

```bash
python scripts/llm_self_tune.py --max-iter 4 --target 0.66 --n-trials 3 \
    --start-attach-off --allow-attach-toggle
```

This is the loop the target market sells: not just generate a 3D scene, but
**autonomously verify and tune the policy that runs on it** — same control
plane, same telemetry, no human in the inner loop.

<details>
<summary>Same loop integrated into the LangGraph reflect node</summary>

The tuning step is also wired in as a `tune_config` node. `reflect_router`
adds a `retry_with_tuned_config` branch that fires when `validate` returns
`success_rate < 1/3` and the tune budget (`_MAX_TUNE_ITERS = 3`) is not
exhausted. The graph loops back to `build_scene → validate → reflect → …`
with the new `SimConfig`.

```bash
three-d-agent --no-external \
  --start-sim-config '{"use_attach_on_close": false}' \
  "put a small red cube on a wooden table"
```

Same convergence behaviour as the standalone script; each `tune_config`
step's reasoning is visible in `messages_log`. Under deliberately weak
starting configs the agent produces sensible proposals (raise friction,
lower mass, widen pregrasp) but doesn't always converge inside the 3-iter
budget — both outcomes are honest: the loop runs, the diagnoses are
interpretable, and the budget caps the cost.

</details>

## Architecture

```
user input
  ├─ planner    (LLM picks ordered steps)
  ├─ decompose  (LLM extracts objects + task + constraints)
  ├─ fetch      (Objaverse retrieve → Hunyuan3D fallback | --no-external uses placeholder cubes)
  ├─ solve_pose (LLM proposes + AABB filter)
  ├─ build      (MuJoCo scene + Franka MJCF from assets/franka/)
  ├─ validate   (damped-LS IK + 4-waypoint pick-and-place + MP4)
  └─ reflect    (retry budget ≤ 2, gates retry_poses / retry_assets / tune_config / success / give_up)
```

Spec and 20-task TDD plan committed before code at `docs/superpowers/specs/`
and `docs/superpowers/plans/`.

## Quickstart

No-GPU path (Windows / Linux / macOS):

```bash
py -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pip install mujoco torch --index-url https://download.pytorch.org/whl/cpu
cp .env.example .env   # then set ANTHROPIC_API_KEY
```

Verify and run:

```bash
.venv/Scripts/python.exe -c "import mujoco; print(mujoco.__version__)"   # 3.9.x
three-d-agent --no-external "put a small red cube on a wooden table"
```

Output:

```json
{
  "scene_id": "scene_790093",
  "video_path": "runs/scene_790093.mp4",
  "success_rate": 1.0,
  "n_success": 3,
  "n_trials": 3
}
```

Open the MP4 in `runs/` to see the trajectory.

<details>
<summary>Full external stack (CUDA GPU + Objaverse-LVIS + Hunyuan3D-2)</summary>

```bash
.venv/Scripts/python.exe -m pip install "git+https://github.com/Tencent/Hunyuan3D-2.git"
.venv/Scripts/python.exe scripts/build_objaverse_index.py
three-d-agent "put a small red cup on a wooden table"
three-d-agent "build the scene in this photo" --image my_photo.png
```

</details>

Tests:

```bash
pytest -m "not e2e"     # CI-safe (unit + integration) — 67 pass
pytest -m e2e           # full pipeline — needs API key (and GPU for external stack)
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Agent framework | LangGraph 1.2 | Explicit state machine — easy to whiteboard |
| LLM | Claude 3.5 Sonnet (default), GPT-4o swappable | Tool-calling stability + JSON reliability |
| Sim | MuJoCo 3.9, CPU | Python 3.14 wheels, no CUDA; same MJCF as Genesis / Isaac |
| Robot | Franka Panda (mujoco_menagerie) | Industry baseline manipulator |
| 3D retrieval | Objaverse-LVIS + FAISS over CLIP-text | Free, ~10M assets |
| 3D generation | Hunyuan3D-2 (lazy import) | SOTA open image-to-3D; deferred when no GPU |
| Validation | Multi-trial pick-and-place + MP4 capture | Quantifiable, recordable |
| Packaging | PEP 621 `pyproject.toml`, hatchling, pytest | Standard, lockfile-free |

## Engineering deep-dives

<details>
<summary><strong>Tested inputs</strong> — 11 prompts run end-to-end with live LLM, all 3/3</summary>

The shape-routing layer is a three-step fallback: explicit `shape_hint` from
`decompose_scene` → keyword match on the description (English + Chinese) →
`mjGEOM_BOX` default. Plus a real GLB-loading branch when
`shape_hint == "mesh"` and a valid file is present.

What was actually run end-to-end with `gsds-claude-sonnet-4-5`, `--no-external`,
recorded in `runs/scene_*.mp4`:

| Prompt | LLM rendered as | Routing path | Pick-and-place |
|---|---|---|---|
| `put a small red cube on a wooden table` | red box | default | 3/3 ✓ |
| `put a blue book on a wooden table` | blue box | default | 3/3 ✓ |
| `put a small green ball on a wooden table` | green sphere | keyword `ball` | 3/3 ✓ |
| `put a small blue cup on a wooden table` | blue cylinder | keyword `cup` | 3/3 ✓ |
| `put a small green watermelon on a wooden table` | green sphere | LLM `shape_hint=sphere` | 3/3 ✓ |
| `put a small purple grape on a wooden table` | purple sphere | LLM `shape_hint=sphere` | 3/3 ✓ |
| `put a yellow pencil on a wooden table` | yellow cylinder | LLM `shape_hint=cylinder` | 3/3 ✓ |
| `桌上放一个红苹果` (Chinese) | red sphere | Chinese keyword `苹果` | 3/3 ✓ |
| `put a small white egg on a wooden table` | white sphere | keyword `egg` | 3/3 ✓ |
| `put a small yellow lemon on a wooden table` | yellow sphere | LLM `shape_hint=sphere` | 3/3 ✓ |
| `put a small wooden brick on a wooden table` | brown box | default | 3/3 ✓ |

11 prompts × 4 colours × 3 primitive shapes × 2 languages, every trial
succeeded. The grasp policy is **shape-agnostic** thanks to attach-on-close,
so adding new object kinds needs no `SimConfig` retuning.

</details>

<details>
<summary><strong>Robot control diagnosis & fix</strong> — two compounding bugs</summary>

Initial implementation had two stacked bugs: the arm started from `qpos=0`
(a singular extended pose IK can't escape via Jacobian descent — 22 cm
tracking error), and each waypoint re-ran IK from scratch so the arm swept
laterally between solutions and knocked the cube sideways before contact.

Fix:

1. Initialise at Franka's `home` keyframe so IK has elbow clearance.
2. Add gravity-feedforward via `mujoco.mj_inverse` so PD doesn't lag under load.
3. Pre-compute IK for all waypoints up front and linearly interpolate
   joint-space between them — smooth descent, no lateral sweep.

End-effector now tracks waypoints to ~1 cm. For the grasp itself I use a
programmatic attach-on-close policy (kinematic weld activated when the
gripper closes) — the standard PyBullet / Isaac demo pattern. Friction-based
grip with Franka's default weak tendon closing torque is its own multi-week
contact-tuning project; the attach-on-close idiom communicates the *Sim2Real*
intent without the rabbit hole. (The self-tuning loop above re-derives a
working friction grasp when the flag is forced off, which is the honest
version.)

</details>

<details>
<summary><strong>Genesis → MuJoCo pivot</strong> — why I switched mid-spec</summary>

Original spec selected Genesis for brand match with the target company.
Pivoted during Phase-0 setup because (a) the dev laptop has no NVIDIA GPU
and (b) no PyBullet wheel exists for Python 3.14. MuJoCo has 3.14 wheels
and consumes the same Franka MJCF format. The LangGraph layer was untouched.

Full rationale at the top of
`docs/superpowers/specs/2026-06-16-3d-agent-design.md`.

</details>

<details>
<summary><strong>Boundaries not yet covered</strong> — where to push in an interview</summary>

Not stoppers for the demo today, but the honest list:

- **`size_hint` is discrete (`small` / `medium` / `large`).** "A cup the size
  of a soda can" needs a continuous `radius_m` field added to `SceneObject`.
- **Task is fixed to `pick_and_place`.** `TaskGoal.action` supports `stack`
  and `push` in the type, but `robot.py` only implements pick-and-place.
- **Mesh-load branch is unit-tested but not visually demoed offline.** Under
  `--no-external`, `fetch_assets` writes placeholder bytes; the builder
  detects this and degrades to the LLM-picked primitive. Real mesh visuals
  need a live Objaverse-LVIS index or Hunyuan3D-2 generator.
- **Cylinder aspect ratio is fixed (radius = 0.7 × half-height).** A pencil
  and a soup can render at the same proportions; one new hint field fixes it.
- **Single manipulable object per scene.** `solve_pose` returns a list and
  the scene builder iterates, but `validate` picks `task.subject` and
  ignores the rest. Multi-object stacking needs a sequence-aware validate.

</details>

## Next steps

In order of impact, what I'd do with another week:

1. **Replace attach-on-close with a true friction grasp** — re-tune fingertip
   pad geometry and friction in `panda.xml` so the gripper holds the cube
   physically. The self-tuning loop already finds a working friction config;
   the next step is making it the default, not the fallback.
2. **Wire a cloud image-to-3D API (Meshy / Tripo)** as a third asset path —
   real AI-generated meshes in the demo with no GPU, ~$0.20 per asset.
3. **Build the Objaverse-LVIS index** (~30 min one-time) to exercise the
   retrieval-first path end-to-end.

