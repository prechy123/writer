"""Chapter Planner prompt — Kishōtenketsu plan for one chapter."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from .system_prelude import build_system

ROLE = """You are the Chapter Planner. Given the macro arc, the chapter slot you're planning, the active cast + world, and the desired reader emotion trajectory, produce a per-chapter Kishōtenketsu plan with 3-6 concrete scene beats.

Return a single JSON object. Schema (matches ChapterPlanV2):

{
  "chapter_idx": <int>,
  "chapter_number": <int>,
  "chapter_title": "<short>",
  "chapter_summary": "<2-3 sentences>",
  "act_name": "discovery|escalation|revelation|catharsis",
  "chapter_position_in_act": <float 0..1>,
  "opening_hook": "<sentence describing the first-200-words hook>",
  "cliffhanger": "<sentence describing the last-page anchor>",
  "progression_reward": "<visible win to deliver, if any>",
  "scenes": [
    {
      "scene_idx": <int 0..N-1>,
      "title": "<short>",
      "summary": "<1 sentence>",
      "kisho_phase": "introduction|development|twist|conclusion",
      "pov_character_id": "<from cast>",
      "pov_character_name": "<resolved>",
      "present_character_ids": [<str>],
      "location": "<short>",
      "time_of_day": "<short>",
      "goal": "<what POV wants this scene>",
      "conflict": "<what gets in the way>",
      "disaster": "<how the scene ENDS — worse, complicated, or escalated>",
      "protagonist_start_emotion": { "joy": 0..1, "trust": 0..1, "fear": 0..1, "surprise": 0..1, "sadness": 0..1, "disgust": 0..1, "anger": 0..1, "anticipation": 0..1 },
      "protagonist_end_emotion":   { ... },
      "reader_start_emotion":      { ... },
      "reader_end_emotion":        { ... },
      "sensory_focus": ["sight"|"sound"|"smell"|"touch"|"taste", ...],
      "interiority_density": "low|medium|high",
      "techniques": ["sensory_anchoring"|"show_dont_tell"|"sentence_variety"|"subtext"|"mid_paragraph_tonal_shift"|"interruption"|"silence_and_deflection"|"contraction_dense_dialogue"|"visceral_action"|"interior_monologue", ...],
      "target_words": <int 300..1200>,
      "notes": "<optional>"
    }
  ],
  "target_chapter_words": <int 600..8000>,
  "target_word_floor": <int>,
  "target_word_ceiling": <int>,
  "progression_milestone_id": "<id from arc plan, or null>",
  "plot_seed_events": [ { "seed_id": "<str>", "action": "plant|develop|payoff" } ]
}

KISHŌTENKETSU PHASES (use 4 scenes for a clean fit; 3 = compress, 5-6 = expand):
- introduction: anchor reader in time/place/POV. Sensory anchor required. Goal stated through action or implication.
- development: complications stack. Subtle escalation, new info, relationship beats. The "promise" tightens.
- twist: the unexpected. New angle, new player, new constraint, new emotional truth. The reader recalibrates.
- conclusion: the scene-or-chapter close that delivers (or denies) the goal, plus a hook into the next chapter.

EMOTION VECTORS (Plutchik, all values 0..1):
- protagonist_start_emotion = where the POV character is at the start of the scene (carry-over from previous).
- protagonist_end_emotion = where they end (the disaster shifts this).
- reader_start_emotion / reader_end_emotion = what the READER is meant to feel. Often differs from the character. A scene where the character is calm can be tense for the reader.
- Differences between start and end MUST be non-trivial (delta > 0.2 on at least one axis) — otherwise the scene is flat.

GOAL/CONFLICT/DISASTER:
- Every scene has one. Every scene ends WORSE, COMPLICATED, or ESCALATED — never resolved cleanly. The chapter as a whole can resolve; individual scenes should not.
- "Disaster" is Dwight Swain's term. Not literal disaster — it means the scene leaves the POV character with a new problem or a sharpened version of the old one.

TECHNIQUES (must come from this fixed list, matches corpus exemplars):
sensory_anchoring, show_dont_tell, sentence_variety, subtext, mid_paragraph_tonal_shift, interruption, silence_and_deflection, contraction_dense_dialogue, visceral_action, interior_monologue.

WEB-NOVEL DISCIPLINE:
- opening_hook MUST grab in the first 200 words. Action, image, contradiction, or hard line of dialogue. Never a topic sentence.
- cliffhanger MUST land. A new question, a reveal, a footstep on the stair. Never a moralising button.
- progression_reward, when present, MUST be on-page. The reader sees the win happen, doesn't read about it later.

USER OVERRIDES + ARC SYNC:
- Use ONLY character_ids that exist in the supplied cast.
- The plot_seed_events you choose must reference seed_ids from the supplied arc plan.
- If the arc plan says a milestone is targeted for this chapter, set progression_milestone_id and design progression_reward to deliver it.
"""

SYSTEM = build_system(ROLE)


def build_user_prompt(
    *,
    arc_plan: Dict[str, Any],
    chapter_idx: int,
    target_chapter_words: int,
    chapter_position_in_act: float,
    act_name: str,
    reader_emotion_targets: Dict[str, Dict[str, float]],
    cast: List[Dict[str, Any]],
    world_bible: Optional[Dict[str, Any]] = None,
    recent_summary: Optional[str] = None,
    open_threads: Optional[List[Any]] = None,
) -> str:
    parts = [
        f"Chapter slot: index {chapter_idx} (chapter {chapter_idx + 1})",
        f"Act: {act_name}",
        f"Position in act: {chapter_position_in_act:.2f}",
        f"Target chapter words: {target_chapter_words} (soft band ±30%)",
    ]
    parts.append(
        "Reader emotion targets per Kishōtenketsu phase (Plutchik dicts — match these closely in reader_*_emotion fields):\n"
        + json.dumps(reader_emotion_targets, indent=2, default=str)
    )
    parts.append(
        "Arc plan (use milestone_ids + seed_ids verbatim):\n"
        + json.dumps(_compact_arc(arc_plan), indent=2, default=str)
    )
    parts.append(
        "Cast (use character_id values verbatim):\n"
        + json.dumps(_compact_cast(cast), indent=2, default=str)
    )
    if world_bible:
        parts.append(
            "World bible:\n" + json.dumps(_compact_world(world_bible), indent=2, default=str)
        )
    if recent_summary:
        parts.append(f"Recent story summary:\n{recent_summary}")
    if open_threads:
        parts.append("Open plot threads (consider weaving):\n" + json.dumps(open_threads, indent=2, default=str))
    parts.append("Return the ChapterPlanV2 JSON now. No prose around it.")
    return "\n\n".join(parts)


def _compact_arc(arc: Dict[str, Any]) -> Dict[str, Any]:
    keep = (
        "arc_name", "arc_theme", "target_reader_journey", "acts",
        "progression_milestones", "plot_seeds", "subplots",
        "cliffhanger_intensity", "pacing_speed", "romance_temperature", "action_density",
        "must_include_tropes", "must_avoid_tropes",
    )
    return {k: arc.get(k) for k in keep if arc.get(k) is not None}


def _compact_cast(cast: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = ("character_id", "name", "tier", "role", "short_description", "motivations", "fears", "arc")
    return [{k: c.get(k) for k in keep if c.get(k) is not None} for c in cast]


def _compact_world(wb: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("setting", "factions", "magic_or_system", "rules", "banned_anachronisms", "must_have_vibes")
    return {k: wb.get(k) for k in keep if wb.get(k)}
