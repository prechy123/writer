"""In-memory cosine similarity over scene-summary embeddings.

Used when Atlas $vectorSearch isn't available. Loads candidate scenes
from Mongo (just the embedding + a minimal projection), scores each
against the query embedding, returns the top-K.

For a 200-scene story at ~768-dim embeddings this is still under a
millisecond — completely fine without a real vector store.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .. import mongo


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def search_scenes(
    *,
    story_id: str,
    query_embedding: List[float],
    k: int = 5,
    exclude_chapter_idx_geq: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return top-K scene summary docs ranked by cosine similarity.

    ``exclude_chapter_idx_geq`` filters out the recent working-memory
    window so episodic retrieval doesn't duplicate working memory.
    """
    if not query_embedding:
        return []
    query: Dict[str, Any] = {"story_id": story_id, "embedding": {"$exists": True}}
    if exclude_chapter_idx_geq is not None:
        query["chapter_idx"] = {"$lt": exclude_chapter_idx_geq}

    cursor = mongo.col(mongo.COL_SCENES).find(
        query,
        projection={
            "chapter_idx": 1,
            "scene_idx": 1,
            "summary": 1,
            "embedding": 1,
            "key_dialogue": 1,
            "mood_snapshot": 1,
        },
    )

    scored: List[tuple[float, Dict[str, Any]]] = []
    for doc in cursor:
        emb = doc.get("embedding") or []
        if not emb:
            continue
        score = _cosine(query_embedding, emb)
        if score > 0.0:
            scored.append((score, doc))
    scored.sort(key=lambda p: p[0], reverse=True)
    return [
        {**doc, "_similarity": round(score, 4), "embedding": None}
        for score, doc in scored[:k]
    ]
