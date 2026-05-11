"""Paste-anything parser — extracts a ParsedSurveyDraft from free text."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..prompts_v2.paste_parser import SYSTEM, build_user_prompt
from ..providers import Router, get_router
from ..schemas_v2 import ParsedSurveyDraft, PastedNotes

logger = logging.getLogger(__name__)


async def parse_pasted_notes(
    notes: PastedNotes,
    *,
    router: Optional[Router] = None,
) -> ParsedSurveyDraft:
    router = router or get_router()
    prompt = build_user_prompt(raw_text=notes.raw_text, hint=notes.hint)

    raw: Dict[str, Any]
    try:
        raw = await router.chat_json(
            role="profiler",  # reuse a roomy long-context role for parsing
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.3,
        )
    except Exception as exc:
        logger.warning("paste_parser: LLM call failed (%s)", exc)
        return ParsedSurveyDraft(notes=[f"Parser unavailable: {exc}"])

    try:
        return ParsedSurveyDraft.model_validate(raw)
    except Exception as exc:
        logger.warning("paste_parser: schema validation failed (%s)", exc)
        return ParsedSurveyDraft(notes=[f"Parser produced invalid output: {exc}"])
