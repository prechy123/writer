"""Edit + regenerate helpers.

Three flows:
    apply_manual_edit         — replace one scene's prose with user-supplied text
    regenerate_single_scene   — re-run the scene_runner for one scene with
                                  optional extra user notes (continuity downstream
                                  is left alone — fast surgical fix)
    cascade_regenerate_from   — wipe everything from (chapter, scene) forward and
                                  re-launch the orchestrator from that point
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional

from .. import mongo
from ..humanisation import humanise
from ..schemas_v2 import (
    CharacterBibleV2,
    SceneBeat,
    StoryStatus,
)
from .orchestrator import launch_story_run
from .scene_runner import run_scene

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manual prose edit (no LLM, but we still run the humaniser to strip slips)
# ---------------------------------------------------------------------------

async def apply_manual_edit(
    *,
    story_id: str,
    chapter_idx: int,
    scene_idx: int,
    new_prose: str,
    run_humaniser: bool = True,
) -> Dict[str, Any]:
    """Replace a scene's prose with user-supplied text. Returns the updated
    scene_drafts_v2 doc."""
    if not new_prose or not new_prose.strip():
        raise ValueError("new_prose must be non-empty")

    existing = mongo.col(mongo.COL_SCENES).find_one(
        {"story_id": story_id, "chapter_idx": chapter_idx, "scene_idx": scene_idx}
    )
    if not existing:
        raise LookupError("scene not found")

    if run_humaniser:
        cast_docs = mongo.list_character_bibles(story_id)
        cast = [
            CharacterBibleV2.model_validate({k: v for k, v in d.items() if k != "_id"})
            for d in cast_docs
        ]
        new_prose, humanise_report = await humanise(new_prose, present_characters=cast)
        humanise_payload = humanise_report.model_dump()
    else:
        humanise_payload = (existing.get("humanisation_report") or {})

    history = list(existing.get("draft_history") or [])
    history.append({
        "prose": existing.get("final_prose") or "",
        "source": "writer_or_editor_previous",
        "recorded_at": datetime.datetime.utcnow(),
    })
    update = {
        "final_prose": new_prose,
        "draft_history": history,
        "humanisation_report": humanise_payload,
        "word_count": len(new_prose.split()),
        "updated_at": datetime.datetime.utcnow(),
        "status": "committed",
        "committed_at": datetime.datetime.utcnow(),
    }
    mongo.col(mongo.COL_SCENES).update_one(
        {"_id": existing["_id"]},
        {"$set": update},
    )

    # Re-stitch the chapter's prose on the story envelope.
    _restitch_chapter(story_id=story_id, chapter_idx=chapter_idx)
    return mongo.col(mongo.COL_SCENES).find_one({"_id": existing["_id"]}) or {}


# ---------------------------------------------------------------------------
# Single-scene regenerate
# ---------------------------------------------------------------------------

async def regenerate_single_scene(
    *,
    story_id: str,
    chapter_idx: int,
    scene_idx: int,
    user_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-run scene_runner.run_scene() for the named scene.

    Continuity DOWNSTREAM (later scenes, ledger, mood snapshots that were
    written from the now-replaced prose) is NOT rolled back. This is the
    fast, surgical path — use ``cascade_regenerate_from`` if you need
    full state replay.
    """
    beat_doc = mongo.col(mongo.COL_BEATS).find_one(
        {"story_id": story_id, "chapter_idx": chapter_idx}
    )
    if not beat_doc:
        raise LookupError("chapter beat sheet not found")
    scenes = beat_doc.get("scenes") or []
    matching = next((s for s in scenes if s.get("scene_idx") == scene_idx), None)
    if not matching:
        raise LookupError("scene_idx not in beat sheet")

    if user_notes:
        merged = dict(matching)
        existing_notes = merged.get("notes") or ""
        merged["notes"] = (existing_notes + "\n\n[USER NOTES]\n" + user_notes).strip()
        beat = SceneBeat.model_validate(merged)
    else:
        beat = SceneBeat.model_validate(matching)

    cast_docs = mongo.list_character_bibles(story_id)
    cast = [
        CharacterBibleV2.model_validate({k: v for k, v in d.items() if k != "_id"})
        for d in cast_docs
    ]

    # Preserve the existing draft as history before overwriting.
    existing_scene = mongo.col(mongo.COL_SCENES).find_one(
        {"story_id": story_id, "chapter_idx": chapter_idx, "scene_idx": scene_idx}
    )
    if existing_scene:
        history = list(existing_scene.get("draft_history") or [])
        history.append({
            "prose": existing_scene.get("final_prose") or "",
            "source": "regenerate_replaced",
            "recorded_at": datetime.datetime.utcnow(),
        })
        mongo.col(mongo.COL_SCENES).update_one(
            {"_id": existing_scene["_id"]},
            {"$set": {"draft_history": history}},
        )

    new_doc = await run_scene(
        story_id=story_id,
        chapter_idx=chapter_idx,
        scene_beat=beat,
        cast=cast,
        arc_seeds=[],  # caller can hand these in via a future signature if needed
    )

    # Re-stitch the chapter's stored prose.
    _restitch_chapter(story_id=story_id, chapter_idx=chapter_idx)
    return new_doc


# ---------------------------------------------------------------------------
# Cascade regenerate
# ---------------------------------------------------------------------------

def cascade_regenerate_from(
    *,
    story_id: str,
    from_chapter_idx: int,
    from_scene_idx: int = 0,
) -> Dict[str, Any]:
    """Wipe state from (from_chapter_idx, from_scene_idx) forward, then
    re-launch the orchestrator from that point.

    NOT async — drops persisted records inline, then kicks off a daemon
    thread. Returns a summary.
    """
    story = mongo.get_story_envelope(story_id)
    if not story:
        raise LookupError("story not found")

    # 1. Drop scene drafts at or after the cutoff
    mongo.col(mongo.COL_SCENES).delete_many({
        "story_id": story_id,
        "$or": [
            {"chapter_idx": {"$gt": from_chapter_idx}},
            {"chapter_idx": from_chapter_idx, "scene_idx": {"$gte": from_scene_idx}},
        ],
    })

    # 2. If wiping mid-chapter, also drop the chapter's beat sheet (it'll re-plan)
    if from_scene_idx == 0:
        mongo.col(mongo.COL_BEATS).delete_many({
            "story_id": story_id,
            "chapter_idx": {"$gte": from_chapter_idx},
        })
    else:
        mongo.col(mongo.COL_BEATS).delete_many({
            "story_id": story_id,
            "chapter_idx": {"$gt": from_chapter_idx},
        })

    # 3. Truncate the chapters[] array on the story envelope
    existing_chapters = list(story.get("chapters") or [])
    kept_chapters = [c for c in existing_chapters if (c.get("chapter_idx") or 0) < from_chapter_idx]
    update: Dict[str, Any] = {
        "chapters": kept_chapters,
        "current_chapter_idx": from_chapter_idx,
        "status": StoryStatus.PENDING.value,
        "updated_at": datetime.datetime.utcnow(),
    }

    # 4. Trim character mood histories down to the cutoff
    for cdoc in mongo.list_character_bibles(story_id):
        hist = cdoc.get("mood_state_history") or []
        kept_hist = [
            h for h in hist
            if (h.get("chapter_idx") or 0) < from_chapter_idx
            or (h.get("chapter_idx") == from_chapter_idx and (h.get("scene_idx") or 0) < from_scene_idx)
        ]
        if len(kept_hist) != len(hist):
            mongo.col(mongo.COL_CHARACTERS).update_one(
                {"_id": cdoc["_id"]},
                {"$set": {"mood_state_history": kept_hist}},
            )

    # 5. Reset open threads / cliffhangers / plot seed events
    # (we keep the arc_plan intact — only execution state resets)
    ledger = story.get("continuity_ledger") or {}
    update["continuity_ledger"] = {
        **ledger,
        "open_threads": [],
        "unresolved_cliffhangers": [],
        "plot_seed_events_log": [],
    }

    mongo.update_story_envelope(story_id, update)

    # 6. Launch from the cutoff
    total = int((story.get("quick_survey") or {}).get("num_chapters") or 0)
    remaining = max(1, total - from_chapter_idx)
    launch_story_run(story_id, batch_size=remaining)

    return {
        "story_id": story_id,
        "rewind_to": {"chapter_idx": from_chapter_idx, "scene_idx": from_scene_idx},
        "remaining_chapters": remaining,
        "status": "queued",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _restitch_chapter(*, story_id: str, chapter_idx: int) -> None:
    """Recompute the chapters[] entry on the story envelope after a scene edit."""
    scenes = list(
        mongo.col(mongo.COL_SCENES)
        .find({"story_id": story_id, "chapter_idx": chapter_idx})
        .sort("scene_idx", 1)
    )
    if not scenes:
        return
    prose = "\n\n".join(s.get("final_prose", "") for s in scenes if s.get("final_prose"))
    word_count = len(prose.split())

    story = mongo.col(mongo.COL_STORIES).find_one(
        {"_id": story_id}, projection={"chapters": 1}
    ) or {}
    chapters = list(story.get("chapters") or [])
    found = False
    for i, c in enumerate(chapters):
        if c.get("chapter_idx") == chapter_idx:
            chapters[i] = {
                **c,
                "text": prose,
                "word_count": word_count,
                "updated_at": datetime.datetime.utcnow(),
            }
            found = True
            break
    if not found:
        chapters.append({
            "chapter_idx": chapter_idx,
            "chapter_number": chapter_idx + 1,
            "text": prose,
            "word_count": word_count,
            "committed_at": datetime.datetime.utcnow(),
        })
    mongo.col(mongo.COL_STORIES).update_one(
        {"_id": story_id},
        {"$set": {"chapters": chapters, "updated_at": datetime.datetime.utcnow()}},
    )
