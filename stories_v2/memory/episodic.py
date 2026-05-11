"""Episodic memory — vector search over prior scene summaries + mood snapshots.

Builds a query embedding from the upcoming scene's beat plan and
retrieves the top-K most similar prior scenes (outside the working
window). Tries Atlas first, falls back to in-memory cosine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .. import mongo
from ..providers import get_router
from . import atlas_vector, inmem_cosine

logger = logging.getLogger(__name__)


def build_query_text(scene_beat: Dict[str, Any]) -> str:
    """Compose a retrieval query from a scene beat plan."""
    parts: List[str] = []
    title = scene_beat.get("title") or ""
    summary = scene_beat.get("summary") or scene_beat.get("intent") or ""
    goal = scene_beat.get("goal") or ""
    conflict = scene_beat.get("conflict") or ""
    disaster = scene_beat.get("disaster") or ""
    pov = scene_beat.get("pov_character_name") or scene_beat.get("pov_character") or ""
    characters = scene_beat.get("present_characters") or []
    if title:
        parts.append(title)
    if summary:
        parts.append(summary)
    if pov:
        parts.append(f"POV: {pov}")
    if goal:
        parts.append(f"Goal: {goal}")
    if conflict:
        parts.append(f"Conflict: {conflict}")
    if disaster:
        parts.append(f"Disaster: {disaster}")
    if characters:
        names = [c if isinstance(c, str) else c.get("name", "") for c in characters]
        names = [n for n in names if n]
        if names:
            parts.append("Characters: " + ", ".join(names))
    return ". ".join(parts)


async def retrieve(
    *,
    story_id: str,
    scene_beat: Dict[str, Any],
    k: int = 5,
    exclude_chapter_idx_geq: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Episodic retrieval. Returns up to K scene summaries."""
    query_text = build_query_text(scene_beat)
    if not query_text:
        return []

    router = get_router()
    embedding = await router.embed(query_text, role="embed")
    if not embedding:
        logger.info("episodic: no embedding available; skipping")
        return []

    if mongo.is_atlas():
        try:
            return atlas_vector.search_scenes(
                story_id=story_id,
                query_embedding=embedding,
                k=k,
                exclude_chapter_idx_geq=exclude_chapter_idx_geq,
            )
        except Exception as exc:
            logger.warning("episodic: Atlas vector search failed (%s); falling back", exc)

    return inmem_cosine.search_scenes(
        story_id=story_id,
        query_embedding=embedding,
        k=k,
        exclude_chapter_idx_geq=exclude_chapter_idx_geq,
    )
