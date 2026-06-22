"""MuJoCo scene builder.

Loads Franka Panda from assets/franka/scene.xml, adds a table and per-object
boxes positioned per their solved poses, compiles, and returns a BuiltScene.
"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco

from three_d_agent.agent.state import SceneObject
from three_d_agent.sim.config import SimConfig


FRANKA_SCENE_XML = Path("assets/franka/scene.xml")
TABLE_HALF_EXTENT = (0.4, 0.4, 0.025)
TABLE_CENTRE_Z = 0.225
TABLE_CENTRE_XY = (0.4, 0.0)

import tempfile as _tempfile

MESH_CACHE_DIR = Path(_tempfile.gettempdir()) / "three_d_agent_mesh_cache"

_SIZE_HALF = {"small": 0.012, "medium": 0.025, "large": 0.05, None: 0.02}

_OBJECT_MASS = 0.02
_OBJECT_FRICTION = (2.0, 0.05, 0.001)


@dataclass
class BuiltScene:
    scene: Any
    franka: dict[str, Any]
    entities: dict[str, int]
    table_top_z: float
    scene_id: str


def build_scene(
    objects: list[SceneObject],
    scene_id: str,
    show_viewer: bool = False,  # noqa: ARG001 - kept for API parity
    sim_config: SimConfig | None = None,
) -> BuiltScene:
    """Build a MuJoCo scene: Franka + table + per-object boxes.

    Parameters
    ----------
    objects:
        List of solved SceneObject instances. Fixed objects (or those without a
        pose) are skipped — only free-moving objects get a body+freejoint.
    scene_id:
        Identifier echoed back on the returned BuiltScene.
    show_viewer:
        Unused for CPU MuJoCo; accepted for API parity with the Genesis stub.
    sim_config:
        Optional `SimConfig` overriding cube geometry/physics constants. NOTE:
        the ``cube_*`` fields (``cube_half``, ``cube_mass``,
        ``cube_friction_slide``) only take effect for objects whose
        ``size_hint == "small"``; non-small size hints continue to use the
        module-level ``_SIZE_HALF`` table and ``_OBJECT_MASS`` /
        ``_OBJECT_FRICTION`` defaults so back-compat with the existing demo
        objects is preserved.
    """
    if not FRANKA_SCENE_XML.exists():
        raise FileNotFoundError(
            f"Franka MJCF not found at {FRANKA_SCENE_XML}. "
            "Re-download via the jsDelivr mirror; see project README."
        )

    spec = mujoco.MjSpec.from_file(str(FRANKA_SCENE_XML))

    table_body = spec.worldbody.add_body(
        name="__table__",
        pos=[TABLE_CENTRE_XY[0], TABLE_CENTRE_XY[1], TABLE_CENTRE_Z],
    )
    table_body.add_geom(
        name="__table_top__",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(TABLE_HALF_EXTENT),
        rgba=[0.55, 0.4, 0.2, 1],
    )

    table_top_z = TABLE_CENTRE_Z + TABLE_HALF_EXTENT[2]

    intended_names: list[str] = []
    for obj in objects:
        if obj.is_fixed or obj.pose is None:
            continue
        if obj.size_hint == "small" and sim_config is not None:
            half = sim_config.cube_half
            mass = sim_config.cube_mass
            friction = (
                sim_config.cube_friction_slide,
                _OBJECT_FRICTION[1],
                _OBJECT_FRICTION[2],
            )
        else:
            half = _SIZE_HALF.get(obj.size_hint, _SIZE_HALF[None])
            mass = _OBJECT_MASS
            friction = _OBJECT_FRICTION
        target_z = max(obj.pose.position[2], table_top_z + half)
        pos = [obj.pose.position[0], obj.pose.position[1], target_z]
        body = spec.worldbody.add_body(name=obj.name, pos=pos)
        body.add_freejoint()

        mesh_loaded = False
        if obj.shape_hint == "mesh" and obj.mesh_path is not None:
            prepared = _try_prepare_mesh(obj.mesh_path, half)
            if prepared is not None:
                mesh_obj_path, scale = prepared
                mesh_name = f"{obj.name}_mesh"
                spec.add_mesh(name=mesh_name,
                              file=str(mesh_obj_path.resolve()),
                              scale=[scale, scale, scale])
                body.add_geom(
                    name=f"{obj.name}_geom",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=mesh_name,
                    rgba=_rgba_from_color(obj.color),
                    mass=mass,
                    friction=list(friction),
                )
                mesh_loaded = True

        if not mesh_loaded:
            geom_type = _geom_type_for(obj)
            body.add_geom(
                name=f"{obj.name}_geom",
                type=geom_type,
                size=_geom_size_for(geom_type, half),
                rgba=_rgba_from_color(obj.color),
                mass=mass,
                friction=list(friction),
            )
        intended_names.append(obj.name)

    model = spec.compile()
    data = mujoco.MjData(model)

    franka = _gather_franka_handles(model)

    entities: dict[str, int] = {}
    for name in intended_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise RuntimeError(f"Body {name!r} not found after compile")
        entities[name] = bid

    return BuiltScene(
        scene=SimpleNamespace(model=model, data=data),
        franka=franka,
        entities=entities,
        table_top_z=table_top_z,
        scene_id=scene_id,
    )


_SPHERE_KEYWORDS = (
    "ball", "sphere", "orange", "apple", "egg", "marble", "tomato",
    "globe", "球", "苹果",
)
_CYLINDER_KEYWORDS = (
    "cup", "can", "bottle", "mug", "cylinder", "tube", "tin", "glass",
    "杯", "瓶", "罐", "筒",
)


_SHAPE_HINT_TO_GEOM = {
    "sphere":   mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "box":      mujoco.mjtGeom.mjGEOM_BOX,
}


def _geom_type_for(obj: SceneObject) -> int:
    """Pick a MuJoCo geom type from the object.

    Priority order (most → least authoritative):
      1. `obj.shape_hint` — set by the `decompose_scene` LLM if present.
      2. Keyword routing on description + name (English + Chinese).
      3. `mjGEOM_BOX` default.
    """
    if obj.shape_hint and obj.shape_hint in _SHAPE_HINT_TO_GEOM:
        return _SHAPE_HINT_TO_GEOM[obj.shape_hint]
    bag = ((obj.description or "") + " " + (obj.name or "")).lower()
    if any(k in bag for k in _SPHERE_KEYWORDS):
        return mujoco.mjtGeom.mjGEOM_SPHERE
    if any(k in bag for k in _CYLINDER_KEYWORDS):
        return mujoco.mjtGeom.mjGEOM_CYLINDER
    return mujoco.mjtGeom.mjGEOM_BOX


def _try_prepare_mesh(mesh_path: Path, target_half_extent: float
                      ) -> tuple[Path, float] | None:
    """Convert a GLB / OBJ source to an OBJ that MuJoCo can read, and compute
    a scale factor that brings its longest axis to ``2 * target_half_extent``.

    Returns ``(obj_path, scale)`` on success, or ``None`` if anything fails
    (file missing, not parseable, empty mesh). Callers should fall back to a
    primitive geom in that case so the demo still runs in offline / dummy
    mode where ``mesh_path`` points at a placeholder.
    """
    if mesh_path is None or not mesh_path.exists():
        return None
    if mesh_path.stat().st_size < 64:
        return None
    try:
        import hashlib

        import trimesh

        digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()[:16]
        MESH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        out = MESH_CACHE_DIR / f"{digest}.obj"
        if not out.exists():
            scene = trimesh.load(mesh_path, force="mesh")
            if scene is None or (hasattr(scene, "vertices")
                                 and len(scene.vertices) == 0):
                return None
            import numpy as np
            rot = trimesh.transformations.rotation_matrix(
                angle=np.pi / 2, direction=[1, 0, 0])
            scene.apply_transform(rot)
            scene.export(out, file_type="obj")
        bbox_mesh = trimesh.load(out, force="mesh")
        if bbox_mesh is None or len(bbox_mesh.vertices) == 0:
            return None
        extents = bbox_mesh.extents
        longest = float(max(extents))
        if longest <= 0:
            return None
        scale = (2 * target_half_extent) / longest
        return out, scale
    except Exception:
        return None


def _geom_size_for(geom_type: int, half: float) -> list[float]:
    """Translate the scalar half-extent into MuJoCo's per-type 3-vector size.

    See MuJoCo docs:
      - mjGEOM_BOX:      (half_x, half_y, half_z)
      - mjGEOM_SPHERE:   (radius, *unused*, *unused*)
      - mjGEOM_CYLINDER: (radius, half_height, *unused*)  -- aligned with Z
    """
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return [half, 0.0, 0.0]
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return [half * 0.7, half, 0.0]
    return [half, half, half]


def _rgba_from_color(color: str | None) -> list[float]:
    table = {
        "red": [0.9, 0.2, 0.2, 1],
        "blue": [0.2, 0.4, 0.9, 1],
        "green": [0.2, 0.8, 0.3, 1],
        "yellow": [0.95, 0.9, 0.2, 1],
        "white": [0.95, 0.95, 0.95, 1],
        "black": [0.15, 0.15, 0.15, 1],
        "brown": [0.55, 0.35, 0.2, 1],
    }
    default = [0.7, 0.7, 0.7, 1]
    if color is None:
        return default
    return table.get(color.lower(), default)


def _gather_franka_handles(model: mujoco.MjModel) -> dict[str, Any]:
    """Look up Franka actuator/joint/body IDs by name.

    The standard panda.xml from MuJoCo Menagerie names arm joints
    ``joint1``..``joint7``, arm actuators ``actuator1``..``actuator7``, the
    gripper actuator ``actuator8`` (single tendon-driven gripper), and the
    end-effector body ``hand``. Verified against the bundled MJCF on
    2026-06-16.
    """

    def name2id(obj_type: int, name: str) -> int:
        i = mujoco.mj_name2id(model, obj_type, name)
        if i < 0:
            raise RuntimeError(f"name {name!r} not in model")
        return i

    arm_joint_ids = [
        name2id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}") for i in range(7)
    ]
    arm_actuator_ids = [
        name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i + 1}") for i in range(7)
    ]
    return {
        "arm_joint_ids": arm_joint_ids,
        "arm_actuator_ids": arm_actuator_ids,
        "hand_body_id": name2id(mujoco.mjtObj.mjOBJ_BODY, "hand"),
        "gripper_actuator_id": name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"),
    }
