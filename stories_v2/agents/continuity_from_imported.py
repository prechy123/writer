"""Continuity-extraction for imported chapters.

When a user imports prior chapters as canon (via the Import wizard), the
engine needs to know:

  - a short summary of each chapter (planner context for chapter N+1)
  - any open threads still unresolved at the end of the last chapter
  - the cliffhanger / hook chapter N ends on
  - the emotional state of each known character after the last chapter

This is a *best-effort* pass — empty fallbacks are acceptable. The caller
should wrap this in try/except and degrade gracefully.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..providers import Router, get_router

logger = logging.getLogger(__name__)


_SYSTEM = """You are a continuity analyst. Given a sequence of prior chapters from a story
that is about to be continued, extract the engine-readable continuity state.

Return a single JSON object with this shape (every field optional, omit if not supported by the text):

{
  "chapter_summaries": [<str>, ...],   // one ~3-sentence summary per chapter, in order
  "open_threads": [<str>, ...],         // plot threads still unresolved at the end of the LAST chapter
  "latest_cliffhanger": "<str>",        // what the last chapter ends on (a hook for chapter N+1)
  "character_moods": {                  // keyed by character name as it appears in the text
    "<name>": {
      "valence": <float between -1 and 1>,
      "arousal": <float between 0 and 1>,
      "summary": "<short note about their state>"
    }
  }
}

RULES:
- Be faithful to what is written. Do not invent threads or moods.
- chapter_summaries must have exactly one entry per input chapter, in order.
- character_moods covers characters who actually appear; omit ones who don't.
- Keep summaries tight (≤ 3 sentences each).
- Output JSON only. No prose around it.
"""


async def extract_continuity_from_imported(
    chapters: List[Dict[str, Any]],
    *,
    known_character_names: Optional[List[str]] = None,
    router: Optional[Router] = None,
) -> Dict[str, Any]:
    """Run an LLM pass to extract continuity state from prior chapters.

    Args:
        chapters: list of ``{"title": str|None, "text": str}`` dicts.
        known_character_names: optional hint listing the cast we already know
            about (helps the LLM normalise names).

    Returns:
        Dict with keys: ``chapter_summaries`` (list[str]), ``open_threads``
        (list[str]), ``latest_cliffhanger`` (str|None), ``character_moods``
        (dict).  Returns empty defaults if the call fails.
    """
    empty: Dict[str, Any] = {
        "chapter_summaries": [""] * len(chapters),
        "open_threads": [],
        "latest_cliffhanger": None,
        "character_moods": {},
    }
    if not chapters:
        return empty

    parts: List[str] = []
    if known_character_names:
        parts.append("Known characters: " + ", ".join(known_character_names))
    for i, ch in enumerate(chapters):
        title = ch.get("title") or f"Chapter {i + 1}"
        text = ch.get("text") or ""
        parts.append(f"--- Chapter {i + 1}: {title} ---\n{text}")
    parts.append(
        f"There are {len(chapters)} chapters above. "
        "Return the continuity JSON now."
    )
    user_prompt = "\n\n".join(parts)

    router = router or get_router()
    try:
        raw = await router.chat_json(
            role="continuity",
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=4000,
            temperature=0.2,
        )
    except Exception as exc:
        logger.warning("continuity_from_imported: LLM call failed (%s)", exc)
        return empty

    if not isinstance(raw, dict):
        return empty

    summaries = raw.get("chapter_summaries") or []
    if not isinstance(summaries, list):
        summaries = []
    # Pad / truncate to match chapter count.
    if len(summaries) < len(chapters):
        summaries = list(summaries) + [""] * (len(chapters) - len(summaries))
    elif len(summaries) > len(chapters):
        summaries = summaries[: len(chapters)]

    open_threads = raw.get("open_threads") or []
    if not isinstance(open_threads, list):
        open_threads = []

    cliffhanger = raw.get("latest_cliffhanger")
    if cliffhanger is not None and not isinstance(cliffhanger, str):
        cliffhanger = str(cliffhanger)

    moods = raw.get("character_moods") or {}
    if not isinstance(moods, dict):
        moods = {}

    return {
        "chapter_summaries": [str(s or "") for s in summaries],
        "open_threads": [str(t) for t in open_threads if t],
        "latest_cliffhanger": cliffhanger,
        "character_moods": moods,
    }
