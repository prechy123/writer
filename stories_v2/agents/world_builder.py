"""World Builder agent — produces a WorldBibleV2 from premise + genre."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

from ..prompts_v2.world_builder import SYSTEM, build_user_prompt
from ..providers import Router, get_router
from ..schemas_v2 import DeepWorld, WorldBibleV2

logger = logging.getLogger(__name__)


async def build_world_bible(
    *,
    story_id: str,
    title: str,
    premise: str,
    genres: Iterable[str],
    tone: Iterable[str],
    user_world: Optional[DeepWorld] = None,
    router: Optional[Router] = None,
) -> WorldBibleV2:
    router = router or get_router()
    user_world_payload = (
        user_world.model_dump(exclude_defaults=True) if user_world else None
    )
    prompt = build_user_prompt(
        title=title,
        premise=premise,
        genres=list(genres),
        tone=list(tone),
        user_world=user_world_payload,
    )

    raw: Dict[str, Any]
    try:
        raw = await router.chat_json(
            role="world_builder",
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=6000,
            temperature=0.5,
        )
    except Exception as exc:
        logger.warning("world_builder: LLM call failed (%s) — using user inputs only", exc)
        raw = user_world_payload or {}

    raw["story_id"] = story_id
    # User overrides win.
    if user_world_payload:
        for key, value in user_world_payload.items():
            if value not in (None, "", [], {}):
                raw[key] = value
    try:
        return WorldBibleV2.model_validate(raw)
    except Exception as exc:
        logger.error("world_builder: schema validation failed (%s); falling back to user-only world", exc)
        merged = {"story_id": story_id, **(user_world_payload or {})}
        return WorldBibleV2.model_validate(merged)
