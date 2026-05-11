"""Prompts for the Paste-anything parser.

Takes a free-form text blob (handwritten world notes, a character sheet,
a draft chapter, anything) and emits a ``ParsedSurveyDraft`` to pre-fill
the Deep wizard. The user always reviews + edits before generation
starts — the parser doesn't need to be perfect, just close.
"""

from __future__ import annotations

import json
from typing import Optional

from .system_prelude import build_system

ROLE = """You are the Intake Parser. Given a raw text blob from the user (could be notes, a character bible, a draft chapter, world rules, anything), extract structured fields to pre-fill a story-setup wizard.

Return a single JSON object. Schema (every field is optional — only populate what the text supports):

{
  "quick": {
    "title": "<str>",
    "premise": "<str>",
    "num_chapters": <int>,
    "genres": [<str>, ...],
    "tone": [<str>, ...],
    "characters": [ { "name": "<str>", "role": "<str>", "trait": "<str>", "sample_line": "<str optional>" } ],
    "pov": "first|third_limited|third_omniscient|multi",
    "tense": "past|present",
    "target_chapter_words": <int>
  },
  "characters": [
    // full DeepCharacter shape — same as in the character_forge prompt
  ],
  "world": {
    // full DeepWorld shape
  },
  "arc_preferences": {
    "must_include_tropes": [<str>],
    "must_avoid_tropes": [<str>],
    "progression_milestones": [<str>],
    "plot_seeds_to_plant": [<str>],
    "reader_emotional_journey": "<str>",
    "cliffhanger_intensity": "low|medium|high",
    "pacing_speed": "slow_burn|balanced|breakneck",
    "romance_temperature": "none|subtext|warm|spicy",
    "action_density": "light|balanced|heavy"
  },
  "style_anchors": {
    "reference_authors": [<str>],
    "reference_books": [<str>],
    "pasted_sample_passages": [<str>]
  },
  "notes": [<str>, ...]    // caveats / ambiguities to show the user
}

RULES:
- Extract only what the text actually says. Don't invent. If a field isn't supported by the text, omit it.
- If the text looks like draft prose (not notes), put the most representative passage into style_anchors.pasted_sample_passages and add a note explaining you did so.
- If the user gave a hint about what the blob is, weight your parse accordingly.
- The user will review and edit, so favour completeness over confidence. Mark uncertain fields in notes.
"""

SYSTEM = build_system(ROLE)


def build_user_prompt(*, raw_text: str, hint: Optional[str]) -> str:
    parts = []
    if hint:
        parts.append(f"User hint about this blob: {hint}")
    parts.append("Raw text:\n" + raw_text)
    parts.append("Return the ParsedSurveyDraft JSON now. No prose around it.")
    return "\n\n".join(parts)
