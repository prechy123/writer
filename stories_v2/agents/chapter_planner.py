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
    continuation_brief: Optional[str] = None,
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
        continuation_brief=continuation_brief,
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

    # Enforce non-empty Goal / Conflict / Disaster. If the planner left
    # any of these blank, the Scene Writer has no scaffolding and fills
    # the void with atmosphere. Try a focused retry first; if that still
    # leaves gaps, derive deterministically from the chapter summary.
    chapter_summary = (raw.get("chapter_summary") or "").strip()
    chapter_title = (raw.get("chapter_title") or f"Chapter {chapter_idx + 1}").strip()
    scenes_needing_fill = [
        sc for sc in scenes_out if not _scene_has_gcd(sc)
    ]
    if scenes_needing_fill:
        try:
            await _fill_missing_gcd(
                scenes=scenes_needing_fill,
                chapter_title=chapter_title,
                chapter_summary=chapter_summary,
                open_threads=open_threads,
                continuation_brief=continuation_brief,
                router=router,
            )
        except Exception as exc:
            logger.warning("chapter_planner: G-C-D refill LLM failed (%s); using deterministic backfill", exc)
        # Deterministic backstop for anything still blank.
        for sc in scenes_out:
            _deterministic_gcd_backfill(
                sc,
                chapter_summary=chapter_summary,
                chapter_title=chapter_title,
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
# G-C-D enforcement
# ---------------------------------------------------------------------------

_GCD_KEYS = ("goal", "conflict", "disaster")
_GCD_REPAIR_FIELDS = ("goal", "conflict", "disaster", "location", "time_of_day")


def _missing_gcd_fields(scene: Dict[str, Any]) -> List[str]:
    """Return the subset of G-C-D / anchor fields that are blank or non-string."""
    missing: List[str] = []
    for key in _GCD_REPAIR_FIELDS:
        v = scene.get(key)
        if not isinstance(v, str) or not v.strip():
            missing.append(key)
    return missing


def _scene_has_gcd(scene: Dict[str, Any]) -> bool:
    """A scene is considered scaffold-complete when goal/conflict/disaster
    AND location are all non-empty. Location anchors the writer in a
    concrete place; without it the writer drifts to mood."""
    for key in _GCD_KEYS:
        v = (scene.get(key) or "").strip() if isinstance(scene.get(key), str) else ""
        if not v:
            return False
    loc = (scene.get("location") or "").strip() if isinstance(scene.get("location"), str) else ""
    if not loc:
        return False
    return True


_GCD_REFILL_SYSTEM = (
    "You are a scene-beat repair tool. The user passes you scene beats "
    "that are missing one or more of: goal, conflict, disaster, location, "
    "time_of_day. Return ONLY the missing fields, never the whole scene. "
    "Be concrete and grounded in the chapter summary. No purple prose."
)


async def _fill_missing_gcd(
    *,
    scenes: List[Dict[str, Any]],
    chapter_title: str,
    chapter_summary: str,
    open_threads: Optional[List[Any]],
    continuation_brief: Optional[str],
    router: Router,
) -> None:
    """Run a single LLM call to refill missing G-C-D / location for any
    scene that came back thin from the planner. Mutates scenes in place.
    """
    if not scenes:
        return

    import json as _json

    payload = {
        "chapter_title": chapter_title,
        "chapter_summary": chapter_summary,
        "open_threads": open_threads or [],
        "continuation_brief": continuation_brief or "",
        "scenes_to_repair": [
            {
                "scene_idx": sc.get("scene_idx"),
                "title": sc.get("title"),
                "summary": sc.get("summary"),
                "kisho_phase": sc.get("kisho_phase"),
                "pov_character_name": sc.get("pov_character_name"),
                "missing": _missing_gcd_fields(sc),
            }
            for sc in scenes
        ],
    }
    user = (
        "=== CHAPTER CONTEXT ===\n"
        + _json.dumps(payload, indent=2, default=str)
        + "\n\n=== TASK ===\n"
        "Return a JSON object: { \"repairs\": [ { \"scene_idx\": <int>, "
        "\"goal\": \"<concrete one-line goal POV wants this scene>\", "
        "\"conflict\": \"<what gets in the way>\", "
        "\"disaster\": \"<how the scene ENDS — worse, complicated, or escalated, NEVER neatly resolved>\", "
        "\"location\": \"<specific physical place>\", "
        "\"time_of_day\": \"<concrete>\" } ] }.\n"
        "Only include fields the scene was missing. Fields must be concrete (no 'somewhere', no 'eventually'). "
        "If you can't ground a field in the chapter summary or continuation brief, invent something specific "
        "that fits the kisho_phase and POV character. NEVER return empty strings."
    )
    raw = await router.chat_json(
        role="chapter_planner",
        system=_GCD_REFILL_SYSTEM,
        messages=[{"role": "user", "content": user}],
        max_tokens=2000,
        temperature=0.4,
        timeout=90.0,
    )
    repairs = (raw or {}).get("repairs") or []
    by_idx = {r.get("scene_idx"): r for r in repairs if isinstance(r, dict)}
    for sc in scenes:
        repair = by_idx.get(sc.get("scene_idx"))
        if not repair:
            continue
        for key in ("goal", "conflict", "disaster", "location", "time_of_day"):
            value = (repair.get(key) or "").strip() if isinstance(repair.get(key), str) else ""
            if value and not (sc.get(key) or "").strip():
                sc[key] = value


def _deterministic_gcd_backfill(
    scene: Dict[str, Any],
    *,
    chapter_summary: str,
    chapter_title: str,
) -> None:
    """Last-resort fill so the writer never sees an empty goal/conflict/disaster.

    The values here are deliberately generic — they're better than empty
    strings (which the writer reads as 'do whatever') but loud enough
    that a human reviewer can spot they came from the fallback.
    """
    phase = (scene.get("kisho_phase") or "").lower()
    pov = (scene.get("pov_character_name") or "the POV character").strip()
    base_summary = chapter_summary or chapter_title

    def _fill(key: str, value: str) -> None:
        cur = scene.get(key)
        cur = cur.strip() if isinstance(cur, str) else ""
        if not cur:
            scene[key] = value

    if phase == "introduction":
        _fill("goal", f"{pov} needs to act on the situation introduced by {base_summary}.")
        _fill("conflict", "The situation resists. New information surfaces that complicates the obvious move.")
        _fill("disaster", "The opening assumption breaks. The shape of the problem is bigger than it looked.")
    elif phase == "development":
        _fill("goal", f"{pov} pushes against the resistance set up in the introduction.")
        _fill("conflict", "An ally pulls a different direction, or a buried fact surfaces.")
        _fill("disaster", "Cost goes up. What worked before stops working.")
    elif phase == "twist":
        _fill("goal", f"{pov} commits to a plan based on what they know.")
        _fill("conflict", "Something they did not see reframes the entire chapter.")
        _fill("disaster", "The reframing erases the safe path. There is no going back to how the chapter started.")
    elif phase == "conclusion":
        _fill("goal", f"{pov} reaches for a resolution to the chapter's central problem.")
        _fill("conflict", "The cost of the resolution is steeper than expected.")
        _fill("disaster", "A new question opens that pulls the next chapter forward.")
    else:
        _fill("goal", f"{pov} acts on the chapter's central problem.")
        _fill("conflict", "The situation pushes back.")
        _fill("disaster", "The scene ends with the problem worse, not better.")

    _fill("location", "(unspecified location — writer should anchor in concrete sensory detail)")
    _fill("time_of_day", "unspecified")


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
        stub: Dict[str, Any] = {
            "kisho_phase": phase,
            "pov_character_name": pov_name,
        }
        _deterministic_gcd_backfill(
            stub,
            chapter_summary=f"chapter {chapter_idx + 1} of the story",
            chapter_title=f"Chapter {chapter_idx + 1}",
        )
        scenes.append(
            SceneBeat(
                scene_idx=i,
                title=f"{phase.title()}",
                summary=f"{phase.title()} of chapter {chapter_idx + 1}.",
                kisho_phase=phase,  # type: ignore[arg-type]
                pov_character_id=pov_id,
                pov_character_name=pov_name,
                present_character_ids=present_ids,
                location=stub.get("location") or "",
                time_of_day=stub.get("time_of_day") or "",
                goal=stub.get("goal") or "",
                conflict=stub.get("conflict") or "",
                disaster=stub.get("disaster") or "",
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
