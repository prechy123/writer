"""Architect prompt — builds the macro 4-act web-novel arc."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from .system_prelude import build_system

ROLE = """You are the Architect. Given a story's premise, world bible, cast, and arc preferences, design a complete macro plan for the requested chapter count.

Return a single JSON object. Schema:

{
  "arc_name": "<str>",
  "arc_theme": "<one line>",
  "target_reader_journey": "<2-3 sentences on what the reader should feel by the end>",
  "acts": [
    { "name": "discovery|escalation|revelation|catharsis", "chapter_range": [<start_idx_inclusive>, <end_idx_inclusive>], "promise": "<str>", "key_beats": [<str>], "notes": "<optional>" }
  ],
  "progression_milestones": [
    { "milestone_id": "<short_kebab>", "name": "<str>", "category": "power|status|romance|wealth|territory|knowledge|skill|team", "description": "<str>", "target_chapter_idx": <int> }
  ],
  "plot_seeds": [
    { "seed_id": "<short_kebab>", "title": "<str>", "summary": "<1-2 sentences>", "planted_chapter_idx": <int>, "payoff_target_chapter_idx": <int>, "status": "planted", "notes": "<optional>" }
  ],
  "subplots": [
    { "subplot_id": "<short_kebab>", "name": "<str>", "type": "romance|rivalry|mentorship|betrayal|mystery|redemption|team|other", "summary": "<str>", "involved_character_ids": [<str>], "start_chapter_idx": <int>, "end_chapter_idx": <int>, "status": "active" }
  ],
  "must_include_tropes": [<str>],
  "must_avoid_tropes": [<str>],
  "cliffhanger_intensity": "low|medium|high",
  "pacing_speed": "slow_burn|balanced|breakneck",
  "romance_temperature": "none|subtext|warm|spicy",
  "action_density": "light|balanced|heavy"
}

ACT RULES (4-act web-novel structure):
- discovery (~25% of chapters): establish character + stakes + world + a clear opening promise. End on the moment that COMMITS the MC to the journey.
- escalation (~35%): rising tension + first significant power-up / level-up / rank-up + at least one major setback. Stakes escalate visibly. New antagonist surface or existing one ramps.
- revelation (~25%): the twist that recontextualises the previous acts. A betrayal, a hidden truth, a power ceiling shattering, an antagonist reveal. Plot seeds start paying off.
- catharsis (~15%): payoffs land. Promised wins delivered. Sequel hook OR genuine arc close, depending on story length.

Chapter index ranges are inclusive on both ends. Cover ALL chapters 0..(num_chapters-1) with no gaps and no overlaps.

PROGRESSION MILESTONES (CRITICAL for web-novel readability):
- For progression genres (cultivation, LitRPG, isekai, system), produce AT LEAST one milestone per 3-4 chapters in escalation + revelation. Each must be a visible, namable win the reader can point to.
- For romance / slice-of-life, milestones can be relationship steps, social wins, mastery markers.
- target_chapter_idx must respect act boundaries (no power-up in chapter 1 of a slow_burn paced story).

PLOT SEEDS:
- Plant 3-7 mysteries / setups. Each gets a payoff_target_chapter_idx. Pay off MOST of them by end; some may carry past this arc.
- Be SPECIFIC. Not "MC has a secret" but "MC's necklace pulses faintly when his rival is nearby; foreshadows bloodline".

USER OVERRIDES:
- If the user provided must_include_tropes / must_avoid_tropes / progression_milestones / plot_seeds_to_plant / reader_emotional_journey: respect them verbatim. Build around them.
- If pacing_speed = breakneck: milestones land more frequently, cliffhangers harder, fewer slow scenes.
- If pacing_speed = slow_burn: milestones spaced wider, more setup, more interiority.
"""

SYSTEM = build_system(ROLE)


def build_user_prompt(
    *,
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
) -> str:
    parts = [
        f"Title: {title}",
        f"Premise:\n{premise}",
        "Genres: " + (", ".join(genres) or "(none)"),
        "Tone: " + (", ".join(tone) or "(none)"),
        f"POV: {pov}",
        f"Chapter count: {num_chapters} (indices 0..{num_chapters - 1})",
    ]
    if continuation_brief and continuation_brief.strip():
        parts.append(
            "AUTHOR'S CONTINUATION BRIEF — what the user wants to happen across these chapters "
            "(use this to anchor milestones, plot_seeds, and the cliffhanger pattern):\n"
            + continuation_brief.strip()
        )
    if world_bible:
        parts.append(
            "World bible (respect rules + factions + system):\n"
            + json.dumps(_compact_world(world_bible), indent=2, default=str)
        )
    if cast:
        parts.append(
            "Cast (use character_id values verbatim in milestones / subplots):\n"
            + json.dumps(_compact_cast(cast), indent=2, default=str)
        )
    if arc_preferences:
        parts.append(
            "User arc preferences (RESPECT verbatim where present):\n"
            + json.dumps(arc_preferences, indent=2, default=str)
        )
    parts.append("Return the ArcPlan JSON now. No prose around it.")
    return "\n\n".join(parts)


def _compact_world(wb: Dict[str, Any]) -> Dict[str, Any]:
    """Drop bulk fields the Architect doesn't need."""
    keep = (
        "setting", "time_period", "technology_level", "social_structure",
        "geography", "factions", "magic_or_system", "rules",
        "banned_anachronisms", "must_have_vibes",
    )
    return {k: wb.get(k) for k in keep if wb.get(k)}


def _compact_cast(cast: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = (
        "character_id", "name", "tier", "role", "short_description",
        "motivations", "fears", "arc", "webnovel_role_hook", "progression_function",
    )
    return [{k: c.get(k) for k in keep if c.get(k) is not None} for c in cast]
