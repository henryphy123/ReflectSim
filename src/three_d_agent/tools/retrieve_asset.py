"""retrieve_asset: query a FAISS index over Objaverse-LVIS captions."""
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss

DEFAULT_INDEX_DIR = Path("src/three_d_agent/assets/index")


@lru_cache(maxsize=1)
def _load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("clip-ViT-B-32")


@lru_cache(maxsize=4)
def _load_index(index_dir: Path) -> tuple[faiss.Index, list[dict]]:
    idx = faiss.read_index(str(index_dir / "lvis.index"))
    meta = [
        json.loads(line)
        for line in (index_dir / "lvis.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return idx, meta


def retrieve_asset(
    object_desc: str,
    top_k: int = 3,
    index_dir: Path | None = None,
    download: bool = True,
) -> list[dict[str, Any]]:
    """Return top-k Objaverse matches ranked by CLIP-text similarity."""
    index_dir = index_dir or DEFAULT_INDEX_DIR
    encoder = _load_encoder()
    idx, meta = _load_index(index_dir)

    q = encoder.encode([object_desc], normalize_embeddings=True).astype("float32")
    scores, ids = idx.search(q, top_k)
    results: list[dict[str, Any]] = []
    for s, i in zip(scores[0], ids[0], strict=False):
        if i < 0:
            continue
        m = meta[int(i)]
        result = {"uid": m["uid"], "caption": m["caption"],
                  "score": float(s), "mesh_path": None}
        if download:
            result["mesh_path"] = _download_one(m["uid"])
        results.append(result)
    return results


def _download_one(uid: str) -> Path | None:
    import objaverse
    paths = objaverse.load_objects(uids=[uid], download_processes=1)
    p = paths.get(uid)
    return Path(p) if p else None
