"""Prompts for the Character Forge agent.

Given a story's premise + world bible + character budget + (optional)
user-supplied character drafts, produce a cast list with full bibles
respecting the budget tiers.

The budget tiers are non-negotiable: the agent gets told the exact
allowed counts and must NOT exceed them. The user can always extend
the cast later via the Deep wizard or direct bible edits.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from .system_prelude import build_system

ROLE = """You are the Character Forge. Produce a cast of characters for a story, respecting a strict tier budget.

Return a single JSON object. Schema:

{{
  "characters": [
    {{
      "name": "<str>",
      "tier": "main|recurring|side",
      "role": "<short, e.g. 'protagonist', 'mentor', 'rival sect head'>",
      "short_description": "<1-2 sentences>",
      "portrait_blurb": "<1-2 sentences, reader-facing on the story page>",
      "age": "<optional, e.g. '17', 'mid-thirties'>",
      "pronouns": "<optional>",
      "background": "<2-4 sentences>",
      "motivations": [<str>, ...],
      "fears": [<str>, ...],
      "secrets": [<str>, ...],
      "arc": "<2-3 sentences on growth trajectory>",
      "education_access": "<concise>",
      "resources_or_limitations": "<concise>",
      "knowledge_sources": [<str>, ...],
      "relationships": [
        {{ "other_name": "<str>", "nature": "ally|rival|mentor|family|romantic|enemy|acquaintance", "tension": <1-10>, "history": "<concise>" }}
      ],
      "voice_fingerprint": {{
        "lexical": {{
          "avg_sentence_words": <float>,
          "sentence_length_stddev": <float>,
          "contraction_rate": <float>,
          "formality": <1-10>,
          "style_register": "<one>",
          "dialect_markers": [<str>, ...],
          "profanity_rate": <float>,
          "hedging_rate": <float>
        }},
        "preferred_phrases": [<str>, ...],
        "banned_phrases": [<str>, ...],
        "verbal_tics": [<str>, ...],
        "catchphrases": [<str>, ...],
        "sample_lines": [<str>, ...],     // 5-8 verbatim lines THIS character could plausibly say
        "silence_style": "<concise>"
      }},
      "webnovel_role_hook": "<concise>",
      "progression_function": "<concise>"
    }}
  ]
}}

BUDGET RULES (HARD):
- The cast must contain between {main_min}-{main_max} MAIN characters, {recurring_min}-{recurring_max} RECURRING, and {side_min}-{side_max} SIDE.
- Main characters are POV-eligible. They MUST have full voice_fingerprint with 5-8 sample_lines.
- Recurring characters appear in 3+ chapters. They MUST have voice_fingerprint with 3-5 sample_lines.
- Side characters appear once or twice. Stub only: name, role, short_description, portrait_blurb, ONE sample_line in voice_fingerprint, no relationships, empty mood tracking.
- NEVER exceed the budget. If you want a bigger cast, you must promote a side character or stay under the ceiling.

VOICE RULES:
- Every main + recurring character MUST have a distinct voice. If two main characters have similar register + formality + dialect_markers, you've failed. Make them sound different.
- sample_lines must sound like that specific character. If a quote could come from any character, rewrite it.
- preferred_phrases and banned_phrases are the most useful fields. Be specific. "compares emotions to weather" not "uses metaphors".

USER OVERRIDES:
- If the user supplied character drafts, use their values verbatim where present. Only fill in missing fields. Never override a user-supplied name, tier, motivation, sample_line, or relationship.
"""


def build_system(main_min: int, main_max: int, recurring_min: int, recurring_max: int, side_min: int, side_max: int) -> str:
    role = ROLE.format(
        main_min=main_min,
        main_max=main_max,
        recurring_min=recurring_min,
        recurring_max=recurring_max,
        side_min=side_min,
        side_max=side_max,
    )
    from .system_prelude import build_system as _bs
    return _bs(role)


def build_user_prompt(
    *,
    title: str,
    premise: str,
    genres: Iterable[str],
    tone: Iterable[str],
    pov: str,
    num_chapters: int,
    world_bible: Optional[Dict[str, Any]] = None,
    user_characters: Optional[List[Dict[str, Any]]] = None,
) -> str:
    parts = [
        f"Title: {title}",
        f"Premise:\n{premise}",
        "Genres: " + (", ".join(genres) or "(none)"),
        "Tone: " + (", ".join(tone) or "(none)"),
        f"POV: {pov}",
        f"Planned chapter count: {num_chapters}",
    ]
    if world_bible:
        parts.append(
            "World bible (already built — characters must fit this world):\n"
            + json.dumps(world_bible, indent=2, default=str)
        )
    if user_characters:
        parts.append(
            "User-supplied character drafts (RESPECT verbatim where present, fill in the rest):\n"
            + json.dumps(user_characters, indent=2, default=str)
        )
    else:
        parts.append("No user-supplied characters. Invent the full cast within the budget.")
    parts.append("Return the cast JSON now. No prose around it.")
    return "\n\n".join(parts)
