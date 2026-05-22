"""Per-chapter loop.

plan_chapter() → persist beat sheet → run each scene → stitch chapter
prose → emit chapter.committed.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional

from .. import mongo
from ..agents import find_act_for_chapter, plan_chapter
from ..schemas_v2 import ArcPlan, CharacterBibleV2
from .progress import emit, update_progress
from .scene_runner import run_scene

logger = logging.getLogger(__name__)


async def run_chapter(
    *,
    story_id: str,
    arc: ArcPlan,
    chapter_idx: int,
    target_chapter_words: int,
    cast: List[CharacterBibleV2],
    world_bible: Optional[Dict[str, Any]] = None,
    recent_summary: Optional[str] = None,
    open_threads: Optional[List[Any]] = None,
    author_profile_hint: Optional[Dict[str, Any]] = None,
    continuation_brief: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one chapter end-to-end. Returns a summary dict."""
    act_name, position_in_act = find_act_for_chapter(arc, chapter_idx)

    await emit(story_id, "chapter.planning", {
        "chapter_idx": chapter_idx, "act": act_name, "position": position_in_act,
    })

    cast_dicts = [c.model_dump() for c in cast]
    plan = await plan_chapter(
        arc=arc,
        chapter_idx=chapter_idx,
        target_chapter_words=target_chapter_words,
        act_name=act_name,
        chapter_position_in_act=position_in_act,
        cast=cast_dicts,
        world_bible=world_bible,
        recent_summary=recent_summary,
        open_threads=open_threads,
        continuation_brief=continuation_brief,
    )

    # Persist beat sheet
    beat_doc = {"_id": f"{story_id}:{chapter_idx}", **plan.model_dump()}
    mongo.col(mongo.COL_BEATS).replace_one(
        {"_id": beat_doc["_id"]}, beat_doc, upsert=True
    )
    await emit(story_id, "chapter.planned", {
        "chapter_idx": chapter_idx,
        "scene_count": len(plan.scenes),
        "title": plan.chapter_title,
        "summary": plan.chapter_summary,
        "opening_hook": plan.opening_hook,
        "cliffhanger": plan.cliffhanger,
    })

    arc_seeds = [s.model_dump() for s in arc.plot_seeds]
    scenes_committed: List[Dict[str, Any]] = []

    for beat in plan.scenes:
        scene_doc = await run_scene(
            story_id=story_id,
            chapter_idx=chapter_idx,
            scene_beat=beat,
            cast=cast,
            arc_seeds=arc_seeds,
            author_profile_hint=author_profile_hint,
            continuation_brief=continuation_brief,
        )
        scenes_committed.append(scene_doc)

    # Stitch chapter prose
    chapter_prose = "\n\n".join(s.get("final_prose", "") for s in scenes_committed if s.get("final_prose"))
    word_count = len(chapter_prose.split())

    mongo.col(mongo.COL_STORIES).update_one(
        {"_id": story_id},
        {
            "$push": {
                "chapters": {
                    "chapter_idx": chapter_idx,
                    "chapter_number": chapter_idx + 1,
                    "title": plan.chapter_title,
                    "summary": plan.chapter_summary,
                    "text": chapter_prose,
                    "word_count": word_count,
                    "committed_at": datetime.datetime.utcnow(),
                }
            },
            "$set": {
                "current_chapter_idx": chapter_idx + 1,
                "updated_at": datetime.datetime.utcnow(),
            },
        },
    )

    await emit(story_id, "chapter.committed", {
        "chapter_idx": chapter_idx,
        "word_count": word_count,
        "scene_count": len(scenes_committed),
        "title": plan.chapter_title,
    })

    return {
        "chapter_idx": chapter_idx,
        "word_count": word_count,
        "scene_count": len(scenes_committed),
        "title": plan.chapter_title,
        "summary": plan.chapter_summary,
    }
