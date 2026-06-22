"""Dummy asset functions for offline / no-external-deps runs.

These let the LangGraph pipeline reach scene_builder without needing
Objaverse + sentence-transformers + Hunyuan3D. The scene_builder ignores
mesh_path anyway and uses coloured boxes per SceneObject, so the dummy
mesh files don't need to be real meshes — they only need to exist on disk.
"""
from pathlib import Path
from typing import Any


DUMMY_CACHE = Path("src/three_d_agent/assets/cache/dummy")


def dummy_retrieve(
    object_desc: str,
    top_k: int = 3,
    **_: Any,
) -> list[dict[str, Any]]:
    """Always returns a single low-score hit so fetch_assets falls back to generate."""
    return [{"uid": f"dummy_{hash(object_desc) & 0xFFFF:x}",
             "caption": object_desc,
             "score": 0.0,
             "mesh_path": None}]


def dummy_generate(
    object_desc: str,
    image_path: Path | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Pretends to generate a 3D asset by touching a placeholder file in the cache."""
    DUMMY_CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{abs(hash(object_desc)) & 0xFFFFFFFF:08x}.glb"
    out = DUMMY_CACHE / key
    if not out.exists():
        out.write_bytes(b"# dummy GLB placeholder\n")
    return {"mesh_path": out, "cached": out.exists(), "used_image": image_path is not None}
