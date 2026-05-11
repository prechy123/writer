"""Top-level orchestrator.

Sequence:
    1. Verify the story envelope + bibles are persisted.
    2. Build or fetch the macro ArcPlan.
    3. Loop over the chapter batch, running run_chapter() per index.
    4. Update final status (awaiting_continue OR completed).

Launched in a daemon thread by the views layer — mirrors v1's
``stories/views.py:_launch_graph`` pattern.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import threading
from typing import Any, Dict, Optional

from .. import mongo
from ..agents import build_arc_plan
from ..schemas_v2 import ArcPlan, CharacterBibleV2, QuickSurvey, StoryStatus
from .chapter_runner import run_chapter
from .progress import emit, update_progress

logger = logging.getLogger(__name__)


async def run_story_async(story_id: str, *, batch_size: Optional[int] = None) -> None:
    """Top-level async runner. Called by ``launch_story_run`` inside a thread."""
    story = mongo.get_story_envelope(story_id)
    if not story:
        logger.error("orchestrator: story %s not found", story_id)
        return

    try:
        await _run_story_inner(story, batch_size=batch_size)
    except Exception as exc:
        logger.exception("orchestrator: story %s failed", story_id)
        update_progress(
            story_id,
            stage="failed",
            message="Generation failed. See server logs.",
            error=str(exc),
            total_chapters=int(story.get("quick_survey", {}).get("num_chapters", 0) or 0),
        )
        await emit(story_id, "story.failed", {"error": f"{type(exc).__name__}: {exc}"})
        mongo.update_story_envelope(story_id, {
            "status": StoryStatus.FAILED.value,
        })


def launch_story_run(story_id: str, *, batch_size: Optional[int] = None) -> None:
    """Launch the story run in a daemon thread with its own event loop.

    Same pattern as v1's stories/views.py:_launch_graph.
    """
    def _target() -> None:
        try:
            asyncio.run(run_story_async(story_id, batch_size=batch_size))
        except Exception:
            logger.exception("orchestrator: thread crashed for %s", story_id)

    t = threading.Thread(target=_target, daemon=True, name=f"v2-story-{story_id[:8]}")
    t.start()
    logger.info("orchestrator: launched daemon for story %s", story_id)


# ---------------------------------------------------------------------------

async def _run_story_inner(story: Dict[str, Any], *, batch_size: Optional[int]) -> None:
    story_id = story["_id"]
    quick = QuickSurvey.model_validate(story["quick_survey"])
    deep = story.get("deep_survey")

    starting_idx = int(story.get("current_chapter_idx") or 0)
    total = int(quick.num_chapters)
    batch_size = int(batch_size or quick.initial_chapters or total)
    end_idx_exclusive = min(total, starting_idx + batch_size)

    await emit(story_id, "story.queued", {
        "starting_chapter_idx": starting_idx,
        "end_chapter_idx_exclusive": end_idx_exclusive,
        "total_chapters": total,
    })
    update_progress(
        story_id,
        stage="architecting",
        message="Building the macro arc plan.",
        percent=2,
        current_chapter=starting_idx + 1,
        completed_chapters=starting_idx,
        total_chapters=total,
    )
    mongo.update_story_envelope(story_id, {"status": StoryStatus.ARCHITECTING.value})

    # ---- Architect (or reuse persisted) ----
    arc = await _get_or_build_arc(story=story, quick=quick, deep=deep, total=total)
    await emit(story_id, "arc.planned", {
        "arc_name": arc.arc_name,
        "act_count": len(arc.acts),
        "milestone_count": len(arc.progression_milestones),
        "plot_seed_count": len(arc.plot_seeds),
    })

    # ---- Cast load ----
    cast_docs = mongo.list_character_bibles(story_id)
    cast = [CharacterBibleV2.model_validate({k: v for k, v in d.items() if k != "_id"}) for d in cast_docs]

    world = mongo.get_world_bible(story_id=story_id) if False else mongo.col(mongo.COL_WORLDS).find_one({"story_id": story_id}, {"_id": 0})

    # ---- Optional author profile ----
    author_profile_hint: Optional[Dict[str, Any]] = None
    pid = quick.profile_id
    if pid:
        prof = mongo.get_profile(pid)
        if prof:
            author_profile_hint = {k: v for k, v in prof.items() if k != "_id"}

    mongo.update_story_envelope(story_id, {"status": StoryStatus.WRITING.value})

    # ---- Chapter batch loop ----
    target_words = int(quick.target_chapter_words or 2200)
    for chapter_idx in range(starting_idx, end_idx_exclusive):
        percent = 5 + int(((chapter_idx + 1 - starting_idx) / max(1, end_idx_exclusive - starting_idx)) * 90)
        update_progress(
            story_id,
            stage="writing",
            message=f"Writing chapter {chapter_idx + 1} of {total}.",
            percent=percent,
            current_chapter=chapter_idx + 1,
            completed_chapters=chapter_idx,
            total_chapters=total,
        )

        # Recent summary = concat of last 2 chapter summaries
        story_now = mongo.get_story_envelope(story_id) or {}
        chapters_so_far = story_now.get("chapters") or []
        recent_summary = " ".join(
            (c.get("summary") or "")
            for c in chapters_so_far[-2:]
            if c.get("summary")
        ) or None
        open_threads = ((story_now.get("continuity_ledger") or {}).get("open_threads") or [])[-12:]

        try:
            await run_chapter(
                story_id=story_id,
                arc=arc,
                chapter_idx=chapter_idx,
                target_chapter_words=target_words,
                cast=cast,
                world_bible=world,
                recent_summary=recent_summary,
                open_threads=open_threads,
                author_profile_hint=author_profile_hint,
            )
        except Exception as exc:
            logger.exception("orchestrator: chapter %s failed for %s", chapter_idx, story_id)
            update_progress(
                story_id,
                stage="failed",
                message=f"Chapter {chapter_idx + 1} generation failed.",
                error=str(exc),
                current_chapter=chapter_idx + 1,
                completed_chapters=chapter_idx,
                total_chapters=total,
            )
            await emit(story_id, "story.failed", {
                "chapter_idx": chapter_idx,
                "error": f"{type(exc).__name__}: {exc}",
            })
            mongo.update_story_envelope(story_id, {"status": StoryStatus.FAILED.value})
            return

    # ---- Terminal status ----
    if end_idx_exclusive >= total:
        update_progress(
            story_id,
            stage="completed",
            message="Story generation complete.",
            percent=100,
            current_chapter=total,
            completed_chapters=total,
            total_chapters=total,
        )
        mongo.update_story_envelope(story_id, {"status": StoryStatus.COMPLETED.value})
        await emit(story_id, "story.completed", {"total_chapters": total})
    else:
        update_progress(
            story_id,
            stage="awaiting_continue",
            message="Batch complete. POST /continue/ to write the next batch.",
            percent=int((end_idx_exclusive / total) * 100),
            current_chapter=end_idx_exclusive,
            completed_chapters=end_idx_exclusive,
            total_chapters=total,
        )
        mongo.update_story_envelope(story_id, {"status": StoryStatus.AWAITING_CONTINUE.value})
        await emit(story_id, "story.batch_done", {
            "completed_chapters": end_idx_exclusive,
            "total_chapters": total,
        })


async def _get_or_build_arc(
    *,
    story: Dict[str, Any],
    quick: QuickSurvey,
    deep: Optional[Dict[str, Any]],
    total: int,
) -> ArcPlan:
    """Return persisted arc if present, else build one and persist."""
    persisted = story.get("arc_plan")
    if isinstance(persisted, dict) and persisted.get("arc_name"):
        try:
            return ArcPlan.model_validate(persisted)
        except Exception:
            logger.warning("orchestrator: persisted arc invalid; rebuilding")

    story_id = story["_id"]
    cast_docs = mongo.list_character_bibles(story_id)
    world = mongo.col(mongo.COL_WORLDS).find_one({"story_id": story_id}, {"_id": 0})

    arc_preferences = None
    if isinstance(deep, dict):
        arc_preferences = deep.get("arc_preferences")

    arc = await build_arc_plan(
        story_id=story_id,
        title=quick.title,
        premise=quick.premise,
        genres=quick.genres,
        tone=quick.tone,
        num_chapters=total,
        pov=quick.pov,
        world_bible=world,
        cast=cast_docs,
        arc_preferences=arc_preferences,
    )
    mongo.update_story_envelope(story_id, {"arc_plan": arc.model_dump()})
    return arc
