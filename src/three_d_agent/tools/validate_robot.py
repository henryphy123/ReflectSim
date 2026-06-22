"""validate_robot: run N pick-and-place trials, aggregate, optionally record."""
from pathlib import Path

from three_d_agent.agent.state import RobotResult, SceneObject, TaskGoal
from three_d_agent.sim.config import SimConfig, TrialTelemetry
from three_d_agent.sim.robot import run_pick_and_place
from three_d_agent.sim.scene_builder import BuiltScene


def _make_frame_capture(built: BuiltScene, path: Path,
                        every_n_steps: int = 10, fps: int = 24):
    """Build a (callback, finalize) pair to record the LIVE pick-and-place
    motion via `run_pick_and_place`'s frame_callback hook. Captures one frame
    every `every_n_steps` sim steps; the resulting MP4 length is approximately
    `(total_sim_steps / every_n_steps) / fps` seconds. With default 2000 steps
    per trial and 10/24, that yields ~8 s of playback — slow enough to perceive
    distinct waypoints (approach, descend, close, lift).
    """
    import mujoco
    import numpy as np

    from three_d_agent.sim.recorder import VideoRecorder

    model = built.scene.model
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.3, 0.0, 0.35]
    cam.distance = 1.4
    cam.azimuth = 50
    cam.elevation = -25

    rec = VideoRecorder(path, fps=fps)
    counter = {"n": 0}

    def callback(data) -> None:
        counter["n"] += 1
        if counter["n"] % every_n_steps != 0:
            return
        renderer.update_scene(data, camera=cam)
        frame = np.asarray(renderer.render(), dtype="uint8")
        rec.add_frame(frame)

    def finalize() -> None:
        rec.close()
        renderer.close()

    return callback, finalize


def validate_robot(
    built: BuiltScene,
    task: TaskGoal,
    objects: list[SceneObject],
    n_trials: int = 3,
    video_dir: Path | None = None,
    sim_config: SimConfig | None = None,
) -> RobotResult:
    failures: list[str] = []
    n_success = 0
    video_path: Path | None = None
    capture_cb = None
    finalize_cb = None

    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{built.scene_id}.mp4"
        try:
            capture_cb, finalize_cb = _make_frame_capture(built, video_path)
        except Exception as e:
            failures.append(f"video_capture_init_failed: {e}")
            video_path = None
            capture_cb = None

    telemetries: list[TrialTelemetry] = []
    for i in range(n_trials):
        cb = capture_cb if (i == n_trials - 1) else None
        outcome, telemetry = run_pick_and_place(
            built, task, sim_config=sim_config, frame_callback=cb
        )
        telemetries.append(telemetry)
        if outcome.get("lifted"):
            n_success += 1
        else:
            failures.append(outcome.get("reason", f"trial_{i}_failed"))

    if finalize_cb is not None:
        try:
            finalize_cb()
        except Exception as e:
            failures.append(f"video_finalize_failed: {e}")
            video_path = None

    return RobotResult(
        n_trials=n_trials,
        n_success=n_success,
        video_path=video_path,
        failure_modes=failures,
        telemetries=telemetries,
    )


def _capture_video(built: BuiltScene, path: Path) -> None:
    """Back-compat shim kept so unit tests that monkeypatch this name still work.
    Real motion recording happens via _make_frame_capture wired into
    run_pick_and_place's frame_callback; this just writes an empty MP4 stub.
    """
    cb, finalize = _make_frame_capture(built, path)
    finalize()
