"""fetch_assets: per-object, retrieve from Objaverse then fall back to generation."""
from pathlib import Path
from typing import Callable, Sequence

from three_d_agent.agent.state import SceneObject

RetrieveFn = Callable[..., list[dict]]
GenerateFn = Callable[..., dict]


def fetch_assets(
    objects: Sequence[SceneObject],
    retrieve: RetrieveFn,
    generate: GenerateFn,
    threshold: float = 0.55,
    image_path: Path | None = None,
) -> list[SceneObject]:
    out: list[SceneObject] = []
    for obj in objects:
        hits = retrieve(obj.description, top_k=3)
        best = max(hits, key=lambda h: h["score"]) if hits else None
        if best and best["score"] >= threshold and best["mesh_path"] is not None:
            out.append(obj.model_copy(update={
                "source": "retrieved",
                "mesh_path": Path(best["mesh_path"]),
            }))
        else:
            result = generate(obj.description, image_path=image_path)
            out.append(obj.model_copy(update={
                "source": "generated",
                "mesh_path": Path(result["mesh_path"]),
            }))
    return out
