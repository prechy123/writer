"""Per-scene loop.

draft → critic panel → editor → optional re-critic → humanise → persist.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any, Dict, List, Optional

from .. import mongo
from ..agents import (
    deepen_scene,
    draft_scene,
    edit_scene,
    should_rerun_critics,
)
from ..agents.continuity_v2 import apply_mood_to_character, refresh_continuity
from ..agents.critics import (
    critique_ai_detect,
    critique_emotion,
    critique_pacing,
    critique_show_dont_tell,
    critique_voice,
)
from ..humanisation import humanise
from ..humanisation.repetition import load_prior_scenes_prose
from ..memory import assemble_context
from ..providers import get_router
from ..schemas_v2 import (
    CharacterBibleV2,
    CriticReport,
    HumanisationReport,
    PlutchikVector,
    SceneBeat,
    SceneDraft,
)
from .progress import emit

logger = logging.getLogger(__name__)

MAX_CYCLES = 2


async def run_scene(
    *,
    story_id: str,
    chapter_idx: int,
    scene_beat: SceneBeat,
    cast: List[CharacterBibleV2],
    arc_seeds: Optional[List[Dict[str, Any]]] = None,
    author_profile_hint: Optional[Dict[str, Any]] = None,
    continuation_brief: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute one scene end-to-end. Returns the persisted SceneDraft dict."""
    scene_idx = scene_beat.scene_idx
    router = get_router()

    await emit(story_id, "scene.started", {
        "chapter_idx": chapter_idx,
        "scene_idx": scene_idx,
        "title": scene_beat.title,
        "pov": scene_beat.pov_character_name,
        "target_words": scene_beat.target_words,
    })

    # 1. Assemble memory context
    memory = await assemble_context(
        story_id=story_id,
        current_chapter_idx=chapter_idx,
        current_scene_idx=scene_idx,
        scene_beat=scene_beat.model_dump(),
        present_character_ids=scene_beat.present_character_ids or [],
        scene_location=scene_beat.location,
    )

    present_chars = [c for c in cast if c.character_id in (scene_beat.present_character_ids or [])]
    if not present_chars:
        present_chars = cast[: min(3, len(cast))]

    # 2. Draft
    try:
        draft = await draft_scene(
            scene_beat=scene_beat,
            memory=memory,
            present_characters=present_chars,
            author_profile_hint=author_profile_hint,
            router=router,
            continuation_brief=continuation_brief,
            chapter_idx=chapter_idx,
        )
    except Exception as exc:
        logger.exception("scene_runner: draft failed (story=%s ch=%s sc=%s)", story_id, chapter_idx, scene_idx)
        await emit(story_id, "scene.failed", {
            "chapter_idx": chapter_idx, "scene_idx": scene_idx,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise

    await emit(story_id, "scene.drafted", {
        "chapter_idx": chapter_idx, "scene_idx": scene_idx,
        "word_count": len(draft.split()),
    })

    critic_reports: List[CriticReport] = []
    final_prose = draft

    # 3. Critic + editor cycles
    for cycle in range(MAX_CYCLES):
        critic_reports = await _run_critic_panel(
            draft=final_prose,
            scene_beat=scene_beat,
            present_characters=present_chars,
        )

        for report in critic_reports:
            await emit(story_id, "scene.critique.report", {
                "chapter_idx": chapter_idx,
                "scene_idx": scene_idx,
                "cycle": cycle,
                "critic": report.critic,
                "score": report.score,
                "finding_count": len(report.findings),
            })

        if not should_rerun_critics(critic_reports):
            break

        if cycle == MAX_CYCLES - 1:
            # Last cycle — accept what we have, don't edit again
            break

        await emit(story_id, "scene.editing", {
            "chapter_idx": chapter_idx, "scene_idx": scene_idx, "cycle": cycle,
        })
        try:
            final_prose = await edit_scene(
                draft=final_prose,
                scene_beat=scene_beat,
                critic_reports=critic_reports,
            )
        except Exception as exc:
            logger.warning("scene_runner: edit failed (%s) — keeping prior draft", exc)
            break

    # 3.5. Deepener pass — runs only when prose is surface-clean but
    # the heuristic detects atmosphere-without-interiority. Skips fast
    # when not needed; high temperature when it does run.
    has_errors = any(
        f.severity == "error" for r in critic_reports for f in r.findings
    )
    pov_character = None
    for c in present_chars:
        if c.character_id == scene_beat.pov_character_id:
            pov_character = c
            break
    final_prose, deepener_report = await deepen_scene(
        prose=final_prose,
        scene_beat=scene_beat,
        pov_character=pov_character,
        present_characters=present_chars,
        overall_critic_score=min((r.score for r in critic_reports), default=1.0),
        has_critic_errors=has_errors,
    )
    if deepener_report.get("ran"):
        await emit(story_id, "scene.deepened", {
            "chapter_idx": chapter_idx, "scene_idx": scene_idx,
            "growth_ratio": deepener_report.get("growth_ratio"),
        })

    # 4. Deterministic humanise — fetch prior committed scenes so the
    # cross-scene repetition detector can strip verbatim recycles.
    await emit(story_id, "scene.humanising", {"chapter_idx": chapter_idx, "scene_idx": scene_idx})
    try:
        prior_scenes = load_prior_scenes_prose(
            story_id, chapter_idx=chapter_idx, scene_idx=scene_idx, lookback=6,
        )
    except Exception:
        prior_scenes = []
    final_prose, humanise_report = await humanise(
        final_prose,
        present_characters=present_chars,
        prior_scenes_prose=prior_scenes,
    )

    # 5. Continuity refresh + mood updates
    continuity = await refresh_continuity(
        story_id=story_id,
        chapter_idx=chapter_idx,
        scene_idx=scene_idx,
        scene_beat=scene_beat,
        final_prose=final_prose,
        cast=[c.model_dump() for c in cast],
        arc_seeds=arc_seeds,
    )

    summary = continuity.get("scene_summary") or ""
    mood_snapshot = continuity.get("protagonist_emotion_end") or PlutchikVector().model_dump()
    reader_emotion_after = continuity.get("reader_emotion_end") or PlutchikVector().model_dump()

    # Apply mood deltas to character bibles
    for delta in continuity.get("character_mood_deltas") or []:
        cid = delta.get("character_id")
        bible_doc = mongo.col(mongo.COL_CHARACTERS).find_one(
            {"story_id": story_id, "character_id": cid}
        )
        if not bible_doc:
            continue
        snapshot = apply_mood_to_character(
            character_bible=bible_doc,
            delta=delta.get("plutchik_delta") or {},
            chapter_idx=chapter_idx,
            scene_idx=scene_idx,
            event_summary=delta.get("last_event_summary") or "",
        )
        mongo.col(mongo.COL_CHARACTERS).update_one(
            {"story_id": story_id, "character_id": cid},
            {
                "$push": {"mood_state_history": snapshot.model_dump()},
                "$set": {"updated_at": datetime.datetime.utcnow()},
            },
        )

    # 6. Embedding for episodic memory
    embedding = None
    try:
        if summary:
            embedding = await router.embed(summary, role="embed")
    except Exception:
        embedding = None

    # 7. Persist SceneDraft
    draft_doc = SceneDraft(
        story_id=story_id,
        chapter_idx=chapter_idx,
        scene_idx=scene_idx,
        draft_prose=draft,
        final_prose=final_prose,
        critic_reports=critic_reports,
        humanisation_report=humanise_report,
        summary=summary,
        key_dialogue=continuity.get("key_dialogue") or [],
        mood_snapshot=mood_snapshot,
        embedding=embedding,
        protagonist_emotion_after=mood_snapshot,
        reader_emotion_after=reader_emotion_after,
        word_count=len(final_prose.split()),
        cycle_count=min(MAX_CYCLES, len(critic_reports) // 5),
        status="committed",
        committed_at=datetime.datetime.utcnow(),
    )
    scene_doc = {"_id": f"{story_id}:{chapter_idx}:{scene_idx}", **draft_doc.model_dump()}
    mongo.col(mongo.COL_SCENES).replace_one(
        {"_id": scene_doc["_id"]},
        scene_doc,
        upsert=True,
    )

    # 8. Merge continuity into story envelope
    _merge_into_ledger(story_id=story_id, continuity=continuity)

    await emit(story_id, "scene.committed", {
        "chapter_idx": chapter_idx, "scene_idx": scene_idx,
        "word_count": draft_doc.word_count,
        "summary": summary,
        "humanisation": humanise_report.model_dump(),
    })

    return scene_doc


# ---------------------------------------------------------------------------

async def _run_critic_panel(
    *,
    draft: str,
    scene_beat: SceneBeat,
    present_characters: List[CharacterBibleV2],
) -> List[CriticReport]:
    beat_dict = scene_beat.model_dump()
    results = await asyncio.gather(
        critique_voice(
            draft=draft, scene_beat=beat_dict, present_characters=present_characters,
        ),
        critique_emotion(draft=draft, scene_beat=beat_dict),
        critique_show_dont_tell(draft=draft, scene_beat=beat_dict),
        critique_ai_detect(draft=draft, scene_beat=beat_dict),
        critique_pacing(draft=draft, scene_beat=beat_dict),
        return_exceptions=True,
    )

    final: List[CriticReport] = []
    for r in results:
        if isinstance(r, CriticReport):
            final.append(r)
        else:
            logger.warning("critic panel: exception %s", r)
    return final


def _merge_into_ledger(*, story_id: str, continuity: Dict[str, Any]) -> None:
    """Merge new threads/seeds into story envelope's continuity_ledger."""
    update: Dict[str, Any] = {}

    new_threads = continuity.get("open_threads_added") or []
    closed = set(continuity.get("open_threads_closed") or [])
    cliffhangers = continuity.get("unresolved_cliffhangers") or []
    plot_events = continuity.get("plot_seed_events") or []
    world_changes = continuity.get("world_state_changes") or []

    # Pull current ledger to merge thread sets cleanly.
    story = mongo.col(mongo.COL_STORIES).find_one(
        {"_id": story_id}, projection={"continuity_ledger": 1}
    )
    ledger = (story or {}).get("continuity_ledger") or {}
    open_threads = [t for t in (ledger.get("open_threads") or []) if t not in closed]
    for t in new_threads:
        if t not in open_threads:
            open_threads.append(t)

    update_payload = {
        "open_threads": open_threads,
        "unresolved_cliffhangers": cliffhangers,
        "world_state_changes_latest": world_changes,
        "plot_seed_events_log": (ledger.get("plot_seed_events_log") or []) + plot_events,
    }
    update["continuity_ledger"] = update_payload
    update["updated_at"] = datetime.datetime.utcnow()

    try:
        mongo.col(mongo.COL_STORIES).update_one({"_id": story_id}, {"$set": update})
    except Exception:
        logger.exception("scene_runner: failed to merge ledger for %s", story_id)
