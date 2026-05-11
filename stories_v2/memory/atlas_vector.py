"""Atlas $vectorSearch episodic retrieval.

Only used when ``mongo.is_atlas()`` returns True (or
``MONGODB_ATLAS_VECTOR=1``). Requires an Atlas Search index named
``scene_drafts_v2_embedding`` over the ``embedding`` field. The index
should be created via the Atlas UI or ``createSearchIndex`` admin command.

If the index does not exist the call will raise an OperationFailure;
``retriever.py`` catches that and falls back to inmem_cosine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .. import mongo

logger = logging.getLogger(__name__)

INDEX_NAME = "scene_drafts_v2_embedding"


def search_scenes(
    *,
    story_id: str,
    query_embedding: List[float],
    k: int = 5,
    exclude_chapter_idx_geq: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not query_embedding:
        return []

    filter_clause: Dict[str, Any] = {"story_id": {"$eq": story_id}}
    if exclude_chapter_idx_geq is not None:
        filter_clause["chapter_idx"] = {"$lt": exclude_chapter_idx_geq}

    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": max(50, k * 10),
                "limit": k,
                "filter": filter_clause,
            }
        },
        {
            "$project": {
                "chapter_idx": 1,
                "scene_idx": 1,
                "summary": 1,
                "key_dialogue": 1,
                "mood_snapshot": 1,
                "_similarity": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    cursor = mongo.col(mongo.COL_SCENES).aggregate(pipeline)
    return list(cursor)
