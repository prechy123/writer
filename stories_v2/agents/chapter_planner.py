"""Chapter Planner agent — produces a ChapterPlanV2 with scene beats."""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from ..corpus import pick_exemplars
from ..emotion import chapter_reader_arc, scene_reader_target
from ..prompts_v2.chapter_planner import SYSTEM, build_user_prompt
from ..providers import Router, get_router
from ..schemas_v2 import ArcPlan, ChapterPlanV2, PlutchikVector, SceneBeat

logger = logging.getLogger(__name__)


async def plan_chapter(
    *,
    arc: ArcPlan,
    chapter_idx: int,
    target_chapter_words: int,
    act_name: str,
    chapter_position_in_act: float,
    cast: List[Dict[str, Any]],
    world_bible: Optional[Dict[str, Any]] = None,
    recent_summary: Optional[str] = None,
    open_threads: Optional[List[Any]] = None,
    router: Optional[Router] = None,
) -> ChapterPlanV2:
    router = router or get_router()

    # Reader emotion arc per phase, used both in the prompt and post-LLM
    # to fill any missing emotion vectors deterministically.
    cliff = arc.cliffhanger_intensity
    reader_arc = chapter_reader_arc(
        act=act_name,
        chapter_position=chapter_position_in_act,
        cliffhanger_intensity=cliff,
    )
    reader_emotion_targets_dump = {
        k: _plutchik_dump(v) for k, v in reader_arc.items()
    }

    prompt = build_user_prompt(
        arc_plan=arc.model_dump(),
        chapter_idx=chapter_idx,
        target_chapter_words=target_chapter_words,
        chapter_position_in_act=chapter_position_in_act,
        act_name=act_name,
        reader_emotion_targets=reader_emotion_targets_dump,
        cast=cast,
        world_bible=world_bible,
        recent_summary=recent_summary,
        open_threads=open_threads,
    )

    raw: Optional[Dict[str, Any]] = None
    try:
        raw = await router.chat_json(
            role="chapter_planner",
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=6000,
            temperature=0.5,
        )
    except Exception as exc:
        logger.warning("chapter_planner: LLM call failed (%s); using fallback", exc)

    if raw is None:
        return _fallback_chapter_plan(
            arc=arc,
            chapter_idx=chapter_idx,
            target_chapter_words=target_chapter_words,
            act_name=act_name,
            cast=cast,
            reader_arc=reader_arc,
        )

    raw["story_id"] = arc.story_id
    raw["chapter_idx"] = chapter_idx
    raw["chapter_number"] = chapter_idx + 1
    raw.setdefault("act_name", act_name)
    raw.setdefault("chapter_position_in_act", chapter_position_in_act)
    raw.setdefault("target_chapter_words", target_chapter_words)
    raw.setdefault("target_word_floor", int(target_chapter_words * 0.70))
    raw.setdefault("target_word_ceiling", int(target_chapter_words * 1.45))

    # Validate scenes; backfill missing reader-emotion vectors from the
    # deterministic arc so the Emotion Critic always has a target.
    scenes_in = raw.get("scenes") or []
    scenes_out: List[Dict[str, Any]] = []
    total = max(1, len(scenes_in))
    for i, sc in enumerate(scenes_in):
        if not isinstance(sc, dict):
            continue
        sc = dict(sc)
        sc["scene_idx"] = i
        if not sc.get("pov_character_id") and cast:
            sc["pov_character_id"] = cast[0].get("character_id", "")
            sc["pov_character_name"] = sc.get("pov_character_name") or cast[0].get("name", "")

        target_reader = scene_reader_target(reader_arc, scene_idx=i, total_scenes=total)
        if not sc.get("reader_start_emotion"):
            sc["reader_start_emotion"] = _plutchik_dump(PlutchikVector())
        if not sc.get("reader_end_emotion"):
            sc["reader_end_emotion"] = _plutchik_dump(target_reader)
        if not sc.get("protagonist_start_emotion"):
            sc["protagonist_start_emotion"] = _plutchik_dump(PlutchikVector(anticipation=0.4))
        if not sc.get("protagonist_end_emotion"):
            sc["protagonist_end_emotion"] = _plutchik_dump(target_reader)
        sc.setdefault("kisho_phase", _phase_for_scene_idx(i, total))
        sc.setdefault("interiority_density", "medium")
        sc.setdefault("target_words", max(300, target_chapter_words // total))

        # Attach corpus exemplars based on the techniques the planner picked.
        techniques = sc.get("techniques") or []
        emotion_tags = _dominant_axes(sc.get("reader_end_emotion") or {})
        genres = (arc.model_extra or {}).get("genres") if hasattr(arc, "model_extra") else []
        try:
            exemplars = pick_exemplars(
                techniques=techniques,
                emotion_tags=emotion_tags,
                pov=None,
                k=3,
            )
            sc["retrieved_exemplar_ids"] = [e.id for e in exemplars]
        except Exception:
            sc["retrieved_exemplar_ids"] = []
        scenes_out.append(sc)

    if not scenes_out:
        # Bail to a deterministic 4-scene Kishōtenketsu shell.
        return _fallback_chapter_plan(
            arc=arc,
            chapter_idx=chapter_idx,
            target_chapter_words=target_chapter_words,
            act_name=act_name,
            cast=cast,
            reader_arc=reader_arc,
        )

    raw["scenes"] = scenes_out
    try:
        return ChapterPlanV2.model_validate(raw)
    except Exception as exc:
        logger.warning("chapter_planner: schema validation failed (%s); using fallback", exc)
        return _fallback_chapter_plan(
            arc=arc,
            chapter_idx=chapter_idx,
            target_chapter_words=target_chapter_words,
            act_name=act_name,
            cast=cast,
            reader_arc=reader_arc,
        )


# ---------------------------------------------------------------------------
# Fallback / helpers
# ---------------------------------------------------------------------------

def _plutchik_dump(v: PlutchikVector) -> Dict[str, float]:
    return v.model_dump()


def _phase_for_scene_idx(i: int, total: int) -> str:
    if total <= 3:
        return ["introduction", "twist", "conclusion"][min(i, 2)]
    ratio = (i + 0.5) / total
    if ratio < 0.25:
        return "introduction"
    if ratio < 0.55:
        return "development"
    if ratio < 0.85:
        return "twist"
    return "conclusion"


def _dominant_axes(plutchik_dict: Dict[str, float], *, top_k: int = 2) -> List[str]:
    if not plutchik_dict:
        return []
    items = sorted(plutchik_dict.items(), key=lambda kv: kv[1] or 0.0, reverse=True)
    return [k for k, v in items[:top_k] if (v or 0) > 0.15]


def _fallback_chapter_plan(
    *,
    arc: ArcPlan,
    chapter_idx: int,
    target_chapter_words: int,
    act_name: str,
    cast: List[Dict[str, Any]],
    reader_arc: Dict[str, PlutchikVector],
) -> ChapterPlanV2:
    """A safe 4-scene Kishōtenketsu skeleton when LLM is unavailable."""
    pov_id = (cast[0].get("character_id") if cast else "") or "unknown"
    pov_name = (cast[0].get("name") if cast else "") or "Protagonist"
    present_ids = [c.get("character_id") for c in cast[:3] if c.get("character_id")]
    per_scene_target = max(300, target_chapter_words // 4)

    scenes: List[SceneBeat] = []
    phases = ["introduction", "development", "twist", "conclusion"]
    for i, phase in enumerate(phases):
        target_reader = reader_arc.get(phase, PlutchikVector())
        scenes.append(
            SceneBeat(
                scene_idx=i,
                title=f"{phase.title()}",
                summary=f"{phase.title()} of chapter {chapter_idx + 1}.",
                kisho_phase=phase,  # type: ignore[arg-type]
                pov_character_id=pov_id,
                pov_character_name=pov_name,
                present_character_ids=present_ids,
                location="",
                time_of_day="",
                goal="",
                conflict="",
                disaster="",
                protagonist_start_emotion=PlutchikVector(anticipation=0.4),
                protagonist_end_emotion=target_reader,
                reader_start_emotion=PlutchikVector(),
                reader_end_emotion=target_reader,
                sensory_focus=random.choice([["sight"], ["sound"], ["touch"], ["smell"]]),  # type: ignore[arg-type]
                interiority_density="medium",
                techniques=["sensory_anchoring", "show_dont_tell"],
                target_words=per_scene_target,
            )
        )

    return ChapterPlanV2(
        story_id=arc.story_id,
        chapter_idx=chapter_idx,
        chapter_number=chapter_idx + 1,
        chapter_title=f"Chapter {chapter_idx + 1}",
        chapter_summary="",
        act_name=act_name,  # type: ignore[arg-type]
        chapter_position_in_act=0.0,
        opening_hook="",
        cliffhanger="",
        progression_reward="",
        scenes=scenes,
        target_chapter_words=target_chapter_words,
        target_word_floor=int(target_chapter_words * 0.70),
        target_word_ceiling=int(target_chapter_words * 1.45),
    )
