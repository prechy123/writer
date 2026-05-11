"""Progress / SSE emission helpers shared by all runners."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, Optional

from .. import mongo
from ..streaming import get_event_bus

logger = logging.getLogger(__name__)


async def emit(
    story_id: str,
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
) -> int:
    """Publish an event onto the SSE bus + persist to run_events_v2."""
    return await get_event_bus().publish(story_id, event_type, payload or {})


def update_progress(
    story_id: str,
    *,
    stage: str,
    message: str = "",
    percent: int = 0,
    current_chapter: Optional[int] = None,
    completed_chapters: int = 0,
    total_chapters: int = 0,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Write a progress event onto the story envelope."""
    event = {
        "stage": stage,
        "message": message,
        "percent": max(0, min(100, int(percent))),
        "current_chapter": current_chapter,
        "completed_chapters": completed_chapters,
        "total_chapters": total_chapters,
        "updated_at": datetime.datetime.utcnow(),
    }
    if error:
        event["error"] = error
    try:
        mongo.col(mongo.COL_STORIES).update_one(
            {"_id": story_id},
            {
                "$set": {"progress": event, "updated_at": datetime.datetime.utcnow()},
                "$push": {"progress_log": {"$each": [event], "$slice": -100}},
            },
        )
    except Exception:
        logger.exception("progress: failed to persist for story %s", story_id)
    return event
