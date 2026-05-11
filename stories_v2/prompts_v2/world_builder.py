"""Prompts for the World Builder agent.

Expands user-provided setting hints (premise + genres + optional Deep
world fields) into a full ``WorldBibleV2`` document. The user may also
provide explicit overrides which the agent must respect verbatim.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from .system_prelude import build_system

ROLE = """You are the World Builder. From the premise, genres, and any user-supplied world inputs, produce a comprehensive worldbuilding bible.

Return a single JSON object. Schema:

{
  "setting": "<one paragraph: time, place, scale, mood>",
  "time_period": "<concise>",
  "technology_level": "<concise>",
  "social_structure": "<concise>",
  "geography": "<concise>",
  "languages": [<str>, ...],
  "factions": [
    { "name": "<str>", "role": "<str>", "goals": [<str>], "methods": [<str>], "notable_members": [<str>], "relationship_to_protagonist": "<str>" }
  ],
  "magic_or_system": {
    "kind": "none|cultivation|litrpg|classical_magic|psionics|tech|other",
    "description": "<str>",
    "progression_path": [<str>, ...],
    "cost_or_drawback": "<str>",
    "hard_limits": [<str>, ...]
  },
  "rules": [
    { "scope": "<str>", "rule": "<str>", "consequence_if_broken": "<str>" }
  ],
  "banned_anachronisms": [<str>, ...],
  "must_have_vibes": [<str>, ...],
  "notes": "<str optional>"
}

RULES:
- If the user provided explicit values for any field, KEEP them verbatim and build around them. Don't override user choices.
- For progression web novels (LitRPG / cultivation / isekai), magic_or_system.progression_path MUST contain at least 5 ordered tiers readers can track (e.g. "Body Tempering → Qi Condensation → Foundation → ...").
- banned_anachronisms is critical: list 5-10 specific concepts the writer must never reference (e.g. "smartphone" in a medieval setting). Be specific.
- must_have_vibes is the atmospheric anchor list: "lantern-lit alleys", "salt wind off the docks", "incense haze in the temple", "stale beer and gunpowder". Concrete sensory tags only.
- factions must have non-trivial goals — not "rule the world" but "secure the western trade route before winter".
"""

SYSTEM = build_system(ROLE)


def build_user_prompt(
    *,
    title: str,
    premise: str,
    genres: Iterable[str],
    tone: Iterable[str],
    user_world: Optional[Dict[str, Any]] = None,
) -> str:
    parts = [
        f"Title: {title}",
        f"Premise:\n{premise}",
        "Genres: " + (", ".join(genres) or "(none specified)"),
        "Tone: " + (", ".join(tone) or "(none specified)"),
    ]
    if user_world:
        parts.append(
            "User-supplied world inputs (RESPECT these verbatim where present):\n"
            + json.dumps(user_world, indent=2, default=str)
        )
    parts.append("Return the WorldBibleV2 JSON now. No prose around it.")
    return "\n\n".join(parts)
