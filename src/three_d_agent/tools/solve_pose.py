"""solve_pose: LLM proposes per-object poses, AABB-overlap filter rejects, LLM retries."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from three_d_agent.agent.json_utils import parse_llm_json
from three_d_agent.agent.prompts import SOLVE_POSE_PROMPT, SYSTEM_PROMPT
from three_d_agent.agent.state import Pose, SceneObject

AABB = tuple[tuple[float, float, float], tuple[float, float, float]]

_SIZE_HALF = {"small": 0.05, "medium": 0.10, "large": 0.20, None: 0.08}


def _aabb_for(obj_pos: tuple[float, float, float], size_hint: str | None) -> AABB:
    h = _SIZE_HALF.get(size_hint, _SIZE_HALF[None])
    return (
        (obj_pos[0] - h, obj_pos[1] - h, obj_pos[2] - h),
        (obj_pos[0] + h, obj_pos[1] + h, obj_pos[2] + h),
    )


def _aabbs_overlap(a: AABB, b: AABB) -> bool:
    for i in range(3):
        if a[1][i] < b[0][i] or b[1][i] < a[0][i]:
            return False
    return True


def _has_collision(
    objects: list[SceneObject],
    poses: list[dict],
) -> bool:
    by_name = {o.name: o for o in objects}
    boxes = []
    for p in poses:
        obj = by_name.get(p["name"])
        if obj is None or obj.is_fixed:
            continue
        boxes.append((p["name"], _aabb_for(tuple(p["position"]), obj.size_hint)))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _aabbs_overlap(boxes[i][1], boxes[j][1]):
                return True
    return False


def _query_llm(
    llm: BaseChatModel,
    objects: list[SceneObject],
    constraints: list[str],
    table_bbox: AABB,
    extra_hint: str = "",
) -> list[dict]:
    obj_list = "\n".join(
        f"- {o.name}, size={o.size_hint or 'medium'}, is_fixed={o.is_fixed}"
        for o in objects
    )
    msg = SOLVE_POSE_PROMPT.format(
        table_bbox=table_bbox,
        object_list=obj_list,
        constraints="\n".join(constraints) or "(none)",
        table_top=table_bbox[1][2],
    )
    if extra_hint:
        msg += f"\n\nAdditional constraint: {extra_hint}"
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                           HumanMessage(content=msg)])
    text = response.content if isinstance(response.content, str) else str(response.content)
    return parse_llm_json(text)


def solve_pose(
    objects: list[SceneObject],
    constraints: list[str],
    table_bbox: AABB,
    llm: BaseChatModel,
    max_attempts: int = 2,
) -> list[SceneObject]:
    poses: list[dict] | None = None
    extra = ""
    for _ in range(max_attempts):
        candidate = _query_llm(llm, objects, constraints, table_bbox, extra)
        if not _has_collision(objects, candidate):
            poses = candidate
            break
        extra = "Previous proposal had overlapping AABBs; separate objects by >= 10 cm."

    if poses is None:
        poses = candidate

    by_name = {p["name"]: p for p in poses}
    out: list[SceneObject] = []
    for o in objects:
        p = by_name.get(o.name)
        if p:
            o = o.model_copy(update={
                "pose": Pose(position=tuple(p["position"]), quat=tuple(p["quat"])),
            })
        out.append(o)
    return out
