"""Build a FAISS index over Objaverse-LVIS captions.

Run once. Writes to src/three_d_agent/assets/index/{lvis.index, lvis.jsonl}.
Use --max-categories to limit size during development.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import faiss
import objaverse


OUT_DIR = Path("src/three_d_agent/assets/index")


@click.command()
@click.option("--max-categories", default=50, show_default=True,
              help="Cap on LVIS categories to include (each contributes ~100 uids).")
@click.option("--per-category", default=30, show_default=True,
              help="Cap on uids per category.")
def main(max_categories: int, per_category: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lvis = objaverse.load_lvis_annotations()
    items: list[dict] = []
    for cat, uids in list(lvis.items())[:max_categories]:
        for uid in uids[:per_category]:
            items.append({"uid": uid, "caption": cat.replace("_", " ")})
    print(f"selected {len(items)} uids from {min(max_categories, len(lvis))} categories")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("clip-ViT-B-32")
    captions = [it["caption"] for it in items]
    embeds = model.encode(captions, normalize_embeddings=True,
                          show_progress_bar=True).astype("float32")

    idx = faiss.IndexFlatIP(embeds.shape[1])
    idx.add(embeds)
    faiss.write_index(idx, str(OUT_DIR / "lvis.index"))
    with (OUT_DIR / "lvis.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"wrote index ({embeds.shape}) and metadata to {OUT_DIR}")


if __name__ == "__main__":
    main()
