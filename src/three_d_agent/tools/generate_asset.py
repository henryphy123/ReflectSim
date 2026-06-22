"""generate_asset: Hunyuan3D-2 wrapper with sha256 caching."""
import hashlib
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel


DEFAULT_CACHE = Path("src/three_d_agent/assets/cache/generated")


class GenerateAssetConfig(BaseModel):
    cache_dir: Path = DEFAULT_CACHE
    model_id: str = "tencent/Hunyuan3D-2"

    model_config = {"protected_namespaces": ()}


def _cache_key(desc: str, image_path: Path | None) -> str:
    h = hashlib.sha256()
    h.update(desc.encode())
    if image_path:
        h.update(b"\0")
        h.update(Path(image_path).read_bytes())
    return h.hexdigest()[:16]


def _build_pipeline(model_id: str):
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_id)


def generate_asset(
    object_desc: str,
    image_path: Path | None,
    cfg: GenerateAssetConfig | None = None,
    pipeline: Any | None = None,
) -> dict[str, Any]:
    cfg = cfg or GenerateAssetConfig()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(object_desc, image_path)
    out = cfg.cache_dir / f"{key}.glb"
    if out.exists():
        return {"mesh_path": out, "cached": True, "used_image": image_path is not None}

    pipe = pipeline if pipeline is not None else _build_pipeline(cfg.model_id)
    img: Image.Image
    if image_path:
        img = Image.open(image_path).convert("RGBA")
    else:
        img = Image.new("RGBA", (256, 256), (200, 200, 200, 255))
    meshes = pipe(image=img)
    if not meshes:
        raise RuntimeError("Hunyuan3D pipeline returned empty mesh list")
    meshes[0].export(str(out))
    return {"mesh_path": out, "cached": False, "used_image": image_path is not None}
