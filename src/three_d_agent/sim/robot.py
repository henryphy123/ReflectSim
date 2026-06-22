"""MuJoCo Franka pick-and-place. Waypoint IK + gravity-feedforward PD control."""
from typing import Any

import mujoco
import numpy as np

from three_d_agent.agent.state import TaskGoal
from three_d_agent.sim.config import SimConfig, TrialTelemetry
from three_d_agent.sim.scene_builder import BuiltScene


_GRIPPER_OPEN = 255.0
_GRIPPER_CLOSED = 0.0
_LIFT_SUCCESS_THRESHOLD_M = 0.05


def _attach_target_to_hand(
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    target_body_id: int,
    hand_body_id: int,
    offset: np.ndarray,
) -> None:
    """Programmatic grasp: while gripper is 'gripping', clamp the target body's
    free-joint qpos to follow the hand. Equivalent to activating a weld
    constraint dynamically — common in PyBullet / Isaac demos. Cleaner than
    fighting contact friction tuning, honest to label as a high-level
    'attach-on-close' policy.
    """
    body_jnt_addr = int(model.body_jntadr[target_body_id])
    qpos_addr = int(model.jnt_qposadr[body_jnt_addr])
    hand_pos = data.xpos[hand_body_id]
    data.qpos[qpos_addr] = hand_pos[0] + offset[0]
    data.qpos[qpos_addr + 1] = hand_pos[1] + offset[1]
    data.qpos[qpos_addr + 2] = hand_pos[2] + offset[2]
    body_dof_addr = int(model.jnt_dofadr[body_jnt_addr])
    data.qvel[body_dof_addr] = 0
    data.qvel[body_dof_addr + 1] = 0
    data.qvel[body_dof_addr + 2] = 0


def run_pick_and_place(
    built: BuiltScene,
    task: TaskGoal,
    sim_config: SimConfig | None = None,
    frame_callback: Any = None,
) -> tuple[dict[str, Any], TrialTelemetry]:
    """Move arm above task.subject, lower, close gripper, lift; report if lifted.

    Uses IK for inverse kinematics and gravity-compensation feedforward on the
    PD position controllers. Without gravity compensation, Franka's default PD
    gains leave ~20 cm of steady-state position error per joint stack, which
    prevents a successful grasp.

    Returns
    -------
    A ``(outcome_dict, telemetry)`` tuple. ``outcome_dict`` keeps its previous
    shape (``lifted``, ``start_z``, ``final_z``, ``reason``) for back-compat
    with ``validate_robot``. ``telemetry`` is a :class:`TrialTelemetry` payload
    used by the LLM self-tuning loop to reason about why a trial failed.
    """
    sim_config = sim_config or SimConfig()
    n_steps = sim_config.n_steps_per_trial
    gripper_pregrasp = sim_config.gripper_pregrasp_ctrl
    tcp = sim_config.tcp_offset_z

    target_body_id = built.entities.get(task.subject)
    if target_body_id is None:
        outcome = {
            "lifted": False,
            "start_z": 0.0,
            "final_z": 0.0,
            "reason": f"subject {task.subject!r} not in scene",
        }
        telemetry = TrialTelemetry(
            success=False,
            success_threshold_m=_LIFT_SUCCESS_THRESHOLD_M,
            cube_start_xyz=(0.0, 0.0, 0.0),
            cube_end_xyz=(0.0, 0.0, 0.0),
            cube_horizontal_displacement_m=0.0,
            cube_vertical_lift_m=0.0,
            final_finger_qpos=0.0,
            final_hand_xyz=(0.0, 0.0, 0.0),
            final_hand_to_cube_distance_m=0.0,
            phase_at_failure="unknown",
            notes=[f"subject {task.subject!r} not in scene"],
        )
        return outcome, telemetry

    model = built.scene.model
    data = built.scene.data
    hand_id = built.franka["hand_body_id"]
    arm_jids = built.franka["arm_joint_ids"]
    arm_aids = built.franka["arm_actuator_ids"]
    grip_aid = built.franka["gripper_actuator_id"]

    arm_qpos_addrs = [int(model.jnt_qposadr[j]) for j in arm_jids]
    arm_dof_addrs = [int(model.jnt_dofadr[j]) for j in arm_jids]
    arm_kps = np.array([model.actuator_gainprm[a, 0] for a in arm_aids])

    _home_arm_qpos = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
    for i, addr in enumerate(arm_qpos_addrs):
        data.qpos[addr] = _home_arm_qpos[i]
        data.ctrl[arm_aids[i]] = _home_arm_qpos[i]
    data.ctrl[grip_aid] = _GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    for _ in range(200):
        mujoco.mj_step(model, data)

    start_pos = data.xpos[target_body_id].copy()
    start_z = float(start_pos[2])

    waypoints = [
        (start_pos + np.array([0.0, 0.0, tcp + 0.20]), _GRIPPER_OPEN,
         sim_config.wp_above_frac),
        (start_pos + np.array([0.0, 0.0, tcp + 0.20]), gripper_pregrasp,
         sim_config.wp_prenarrow_frac),
        (start_pos + np.array([0.0, 0.0, tcp + 0.00]), gripper_pregrasp,
         sim_config.wp_descend_frac),
        (start_pos + np.array([0.0, 0.0, tcp + 0.00]), _GRIPPER_CLOSED,
         sim_config.wp_close_frac),
        (start_pos + np.array([0.0, 0.0, tcp + 0.20]), _GRIPPER_CLOSED,
         sim_config.wp_lift_frac),
    ]

    saved_qpos = data.qpos.copy()
    target_qs: list[np.ndarray] = []
    for target_xyz, _, _ in waypoints:
        target_qs.append(
            _ik_position(model, data, np.asarray(target_xyz),
                         hand_id, arm_qpos_addrs)
        )
    data.qpos[:] = saved_qpos

    cur_arm = np.array([data.qpos[a] for a in arm_qpos_addrs])

    grasp_offset: np.ndarray | None = None

    for wp_idx, ((target_xyz, grip_ctrl, frac), target_arm) in enumerate(
        zip(waypoints, target_qs, strict=True)
    ):
        attaching_starts_here = (
            wp_idx == 3 and sim_config.use_attach_on_close
        )
        steps_this = max(1, int(n_steps * frac))
        for k in range(steps_this):
            alpha = (k + 1) / steps_this
            interp = cur_arm + alpha * (target_arm - cur_arm)
            grav_tau = _gravity_torques_at(
                model, data, interp, arm_qpos_addrs, arm_dof_addrs
            )
            ctrl_offset = grav_tau / arm_kps
            for i, act_id in enumerate(arm_aids):
                lo, hi = model.actuator_ctrlrange[act_id]
                data.ctrl[act_id] = float(np.clip(interp[i] + ctrl_offset[i],
                                                  lo, hi))
            data.ctrl[grip_aid] = grip_ctrl
            mujoco.mj_step(model, data)
            if attaching_starts_here and grasp_offset is None and k == steps_this - 1:
                grasp_offset = data.xpos[target_body_id] - data.xpos[hand_id]
            if grasp_offset is not None:
                _attach_target_to_hand(model, data, target_body_id, hand_id,
                                       grasp_offset)
                mujoco.mj_forward(model, data)
            if frame_callback is not None:
                frame_callback(data)
        cur_arm = target_arm.copy()

    final_pos = data.xpos[target_body_id].copy()
    final_z = float(final_pos[2])
    lifted = (final_z - start_z) > _LIFT_SUCCESS_THRESHOLD_M
    outcome = {
        "lifted": lifted,
        "start_z": start_z,
        "final_z": final_z,
        "reason": None if lifted else "object did not lift > 5 cm",
    }

    cube_start_xyz = (float(start_pos[0]), float(start_pos[1]), float(start_pos[2]))
    cube_end_xyz = (float(final_pos[0]), float(final_pos[1]), float(final_pos[2]))
    dx = cube_end_xyz[0] - cube_start_xyz[0]
    dy = cube_end_xyz[1] - cube_start_xyz[1]
    horizontal_disp = float(np.sqrt(dx * dx + dy * dy))
    vertical_lift = cube_end_xyz[2] - cube_start_xyz[2]

    finger_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    if finger_jid >= 0:
        final_finger_qpos = float(data.qpos[int(model.jnt_qposadr[finger_jid])])
    else:
        final_finger_qpos = 0.0

    hand_pos = data.xpos[hand_id]
    final_hand_xyz = (float(hand_pos[0]), float(hand_pos[1]), float(hand_pos[2]))
    hcx = cube_end_xyz[0] - final_hand_xyz[0]
    hcy = cube_end_xyz[1] - final_hand_xyz[1]
    hcz = cube_end_xyz[2] - final_hand_xyz[2]
    hand_to_cube = float(np.sqrt(hcx * hcx + hcy * hcy + hcz * hcz))

    if lifted:
        phase_at_failure: str = "none"
    elif horizontal_disp > 0.02:
        phase_at_failure = "descend"
    elif final_finger_qpos > 0.03:
        phase_at_failure = "close"
    elif hand_to_cube > 0.10:
        phase_at_failure = "lift"
    else:
        phase_at_failure = "unknown"

    notes = [
        f"cube moved {horizontal_disp * 100:.1f} cm horizontally",
        f"final finger qpos {final_finger_qpos:.4f} (0=closed)",
        f"hand-cube gap {hand_to_cube * 100:.1f} cm",
    ]

    telemetry = TrialTelemetry(
        success=lifted,
        success_threshold_m=_LIFT_SUCCESS_THRESHOLD_M,
        cube_start_xyz=cube_start_xyz,
        cube_end_xyz=cube_end_xyz,
        cube_horizontal_displacement_m=horizontal_disp,
        cube_vertical_lift_m=float(vertical_lift),
        final_finger_qpos=final_finger_qpos,
        final_hand_xyz=final_hand_xyz,
        final_hand_to_cube_distance_m=hand_to_cube,
        phase_at_failure=phase_at_failure,
        notes=notes,
    )
    return outcome, telemetry


def _ik_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    hand_body_id: int,
    arm_qpos_addrs: list[int],
    max_iter: int = 200,
    tol: float = 1e-3,
    step: float = 0.5,
    damping: float = 0.1,
) -> np.ndarray:
    """Damped least-squares IK. Returns final 7-vector of arm qpos.

    Mutates data temporarily -- caller should save/restore data.qpos if needed.
    """
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        cur = data.xpos[hand_body_id]
        err = target_pos - cur
        if np.linalg.norm(err) < tol:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hand_body_id)
        J = jacp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
        for i, addr in enumerate(arm_qpos_addrs):
            data.qpos[addr] += step * dq[i]
    return np.array([data.qpos[addr] for addr in arm_qpos_addrs])


def _gravity_torques_at(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_arm_qpos: np.ndarray,
    arm_qpos_addrs: list[int],
    arm_dof_addrs: list[int],
) -> np.ndarray:
    """Compute the joint torques (7,) needed to hold the arm at target_arm_qpos
    against gravity (acceleration = 0). Uses mj_inverse with zero qvel/qacc.

    This mutates data.qpos/qvel/qacc and restores them.
    """
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_qacc = data.qacc.copy()

    for i, addr in enumerate(arm_qpos_addrs):
        data.qpos[addr] = target_arm_qpos[i]
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    mujoco.mj_inverse(model, data)
    tau = np.array([data.qfrc_inverse[dof] for dof in arm_dof_addrs])

    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    data.qacc[:] = saved_qacc
    return tau
