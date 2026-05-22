"""Architect agent — produces an ArcPlan."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from ..prompts_v2.architect import SYSTEM, build_user_prompt
from ..providers import Router, get_router
from ..schemas_v2 import ArcPlan

logger = logging.getLogger(__name__)


async def build_arc_plan(
    *,
    story_id: str,
    title: str,
    premise: str,
    genres: Iterable[str],
    tone: Iterable[str],
    num_chapters: int,
    pov: str,
    world_bible: Optional[Dict[str, Any]] = None,
    cast: Optional[List[Dict[str, Any]]] = None,
    arc_preferences: Optional[Dict[str, Any]] = None,
    continuation_brief: Optional[str] = None,
    router: Optional[Router] = None,
) -> ArcPlan:
    router = router or get_router()
    prompt = build_user_prompt(
        title=title,
        premise=premise,
        genres=list(genres),
        tone=list(tone),
        num_chapters=int(num_chapters),
        pov=pov,
        world_bible=world_bible,
        cast=cast,
        arc_preferences=arc_preferences,
        continuation_brief=continuation_brief,
    )

    try:
        raw = await router.chat_json(
            role="architect",
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.5,
        )
    except Exception as exc:
        logger.warning("architect: LLM call failed (%s) — building a fallback skeleton", exc)
        return _fallback_arc(story_id=story_id, num_chapters=num_chapters, arc_preferences=arc_preferences)

    raw["story_id"] = story_id
    raw["num_chapters"] = int(num_chapters)

    try:
        return ArcPlan.model_validate(raw)
    except Exception as exc:
        logger.warning("architect: schema validation failed (%s); using fallback", exc)
        return _fallback_arc(story_id=story_id, num_chapters=num_chapters, arc_preferences=arc_preferences)


def _fallback_arc(
    *,
    story_id: str,
    num_chapters: int,
    arc_preferences: Optional[Dict[str, Any]],
) -> ArcPlan:
    """A safe skeleton arc when the LLM is unavailable.

    Divides the chapter count into the canonical 25/35/25/15 split so
    the Chapter Planner has something to anchor against. Milestones and
    plot seeds are left empty — Continuity v2 can add them on the fly.
    """
    n = max(1, int(num_chapters))
    end_discovery = max(0, int(round(n * 0.25)) - 1)
    end_escalation = max(end_discovery + 1, int(round(n * 0.60)) - 1)
    end_revelation = max(end_escalation + 1, int(round(n * 0.85)) - 1)
    end_catharsis = n - 1

    prefs = arc_preferences or {}
    return ArcPlan(
        story_id=story_id,
        arc_name="Primary Arc",
        arc_theme=prefs.get("reader_emotional_journey", "") or "",
        target_reader_journey=prefs.get("reader_emotional_journey", "") or "",
        num_chapters=n,
        acts=[
            {"name": "discovery", "chapter_range": (0, end_discovery), "promise": "Establish character and stakes.", "key_beats": []},
            {"name": "escalation", "chapter_range": (end_discovery + 1, end_escalation), "promise": "Raise stakes; deliver first wins.", "key_beats": []},
            {"name": "revelation", "chapter_range": (end_escalation + 1, end_revelation), "promise": "Twist + recontextualisation.", "key_beats": []},
            {"name": "catharsis", "chapter_range": (end_revelation + 1, end_catharsis), "promise": "Payoffs land; arc closes.", "key_beats": []},
        ],
        must_include_tropes=prefs.get("must_include_tropes", []) or [],
        must_avoid_tropes=prefs.get("must_avoid_tropes", []) or [],
        cliffhanger_intensity=prefs.get("cliffhanger_intensity", "high"),
        pacing_speed=prefs.get("pacing_speed", "balanced"),
        romance_temperature=prefs.get("romance_temperature", "subtext"),
        action_density=prefs.get("action_density", "balanced"),
    )


def find_act_for_chapter(arc: ArcPlan, chapter_idx: int) -> tuple[str, float]:
    """Return ``(act_name, position_in_act_0_to_1)`` for a chapter index."""
    for act in arc.acts:
        start, end = act.chapter_range
        if start <= chapter_idx <= end:
            span = max(1, end - start + 1)
            pos = (chapter_idx - start) / span
            return act.name, float(pos)
    # Fallback: past the planned acts → final act
    if arc.acts:
        return arc.acts[-1].name, 1.0
    return "discovery", 0.0
