"""Continuity v2 prompt — refresh character mood + ledger after a scene."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .system_prelude import build_system

ROLE = """You are the Continuity Refresher. A scene just committed. Read the prose, the scene beat, the current ledger, and produce a STRUCTURED update.

Return one JSON object:

{
  "scene_summary": "<2-4 sentences plot-level summary, no fluff>",
  "key_dialogue": [<str>, ...],                       // 1-3 stand-out lines worth caching for episodic memory
  "character_mood_deltas": [
    {
      "character_id": "<from cast>",
      "name": "<resolved>",
      "plutchik_delta": { "joy": -0.2..+0.2, "trust": ..., "fear": ..., "surprise": ..., "sadness": ..., "disgust": ..., "anger": ..., "anticipation": ... },
      "last_event_summary": "<1 sentence: what happened to this character this scene>"
    }
  ],
  "protagonist_emotion_end": { "joy": 0..1, "trust": 0..1, "fear": 0..1, "surprise": 0..1, "sadness": 0..1, "disgust": 0..1, "anger": 0..1, "anticipation": 0..1 },
  "reader_emotion_end": { ... same shape ... },
  "world_state_changes": [<str>, ...],
  "plot_seed_events": [
    { "seed_id": "<from arc plan>", "action": "developed|paid_off" }
  ],
  "open_threads_added": [<str>, ...],
  "open_threads_closed": [<str>, ...],
  "unresolved_cliffhangers": [<str>, ...]
}

RULES:
- Use character_id values verbatim from the cast list provided.
- plutchik_delta values are SIGNED, range -0.2 to +0.2. They are added to the character's current vector (clamped 0..1 by the engine).
- protagonist_emotion_end and reader_emotion_end are ABSOLUTE end-states (0..1 each axis), not deltas.
- Only emit plot_seed_events for seeds the prose actually engages with. Don't speculate.
- Be precise. No flavour. The engine reads this as structured data."""

SYSTEM = build_system(ROLE)


def build_user_prompt(
    *,
    scene_beat: Dict[str, Any],
    final_prose: str,
    cast: List[Dict[str, Any]],
    arc_seeds: Optional[List[Dict[str, Any]]] = None,
    current_ledger: Optional[Dict[str, Any]] = None,
) -> str:
    parts = ["=== SCENE BEAT ===\n" + json.dumps(_compact_beat(scene_beat), indent=2, default=str)]
    parts.append("=== FINAL PROSE ===\n" + (final_prose or ""))
    parts.append("=== CAST (use character_id verbatim) ===\n" + json.dumps(_compact_cast(cast), indent=2, default=str))
    if arc_seeds:
        parts.append("=== ARC PLOT SEEDS ===\n" + json.dumps(arc_seeds, indent=2, default=str))
    if current_ledger:
        parts.append("=== CURRENT LEDGER (for delta context) ===\n" + json.dumps(current_ledger, indent=2, default=str))
    parts.append("=== RETURN THE CONTINUITY JSON NOW ===")
    return "\n\n".join(parts)


def _compact_beat(beat: Dict[str, Any]) -> Dict[str, Any]:
    keep = (
        "scene_idx", "title", "summary", "kisho_phase",
        "pov_character_id", "pov_character_name", "present_character_ids",
        "location", "time_of_day", "goal", "conflict", "disaster",
    )
    return {k: beat.get(k) for k in keep if beat.get(k) is not None}


def _compact_cast(cast: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"character_id": c.get("character_id"), "name": c.get("name"), "tier": c.get("tier")}
        for c in cast
        if c.get("character_id")
    ]
