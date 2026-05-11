"""Character Forge — builds the cast with strict tier-budget enforcement.

The budget table lives in ``schemas_v2.common.character_budget_for``.
The agent prompt is given the per-tier (min, max) counts, and the
post-LLM validator clamps any deviation rather than failing.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional

from ..prompts_v2.character_forge import build_system, build_user_prompt
from ..providers import Router, get_router
from ..schemas_v2 import (
    CharacterBibleV2,
    CharacterTier,
    DeepCharacter,
    VoiceFingerprint,
    character_budget_for,
)

logger = logging.getLogger(__name__)


async def forge_cast(
    *,
    story_id: str,
    title: str,
    premise: str,
    genres: Iterable[str],
    tone: Iterable[str],
    pov: str,
    num_chapters: int,
    world_bible: Optional[Dict[str, Any]] = None,
    user_characters: Optional[List[DeepCharacter]] = None,
    router: Optional[Router] = None,
) -> List[CharacterBibleV2]:
    """Produce a tier-budgeted cast for a story.

    Returns a list of CharacterBibleV2. Caller persists each to
    ``character_bibles_v2`` and references the cast from the story envelope.
    """
    router = router or get_router()
    budget = character_budget_for(int(num_chapters))
    main_min, main_max = budget["main"]
    rec_min, rec_max = budget["recurring"]
    side_min, side_max = budget["side"]

    system = build_system(main_min, main_max, rec_min, rec_max, side_min, side_max)
    user_chars_payload = (
        [c.model_dump(exclude_defaults=True) for c in (user_characters or [])]
        or None
    )
    prompt = build_user_prompt(
        title=title,
        premise=premise,
        genres=list(genres),
        tone=list(tone),
        pov=pov,
        num_chapters=int(num_chapters),
        world_bible=world_bible,
        user_characters=user_chars_payload,
    )

    raw: Dict[str, Any]
    try:
        raw = await router.chat_json(
            role="character_forge",
            system=system,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.6,
        )
    except Exception as exc:
        logger.warning("character_forge: LLM call failed (%s) — using user-supplied cast only", exc)
        raw = {"characters": user_chars_payload or []}

    raw_chars = raw.get("characters") or []
    bibles = _to_bibles(story_id=story_id, raw_chars=raw_chars)
    bibles = _enforce_budget(bibles, budget)
    return bibles


def _to_bibles(*, story_id: str, raw_chars: List[Dict[str, Any]]) -> List[CharacterBibleV2]:
    out: List[CharacterBibleV2] = []
    for entry in raw_chars:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        payload: Dict[str, Any] = {**entry}
        payload["story_id"] = story_id
        payload.setdefault("character_id", str(uuid.uuid4()))
        # Normalise tier
        tier = payload.get("tier") or "recurring"
        try:
            payload["tier"] = CharacterTier(str(tier).lower()).value
        except ValueError:
            payload["tier"] = CharacterTier.RECURRING.value
        # Normalise voice fingerprint (validate or drop)
        vf = payload.get("voice_fingerprint")
        if vf is not None:
            try:
                payload["voice_fingerprint"] = VoiceFingerprint.model_validate(vf).model_dump()
            except Exception:
                payload["voice_fingerprint"] = None
        # Coerce
        try:
            out.append(CharacterBibleV2.model_validate(payload))
        except Exception as exc:
            logger.warning("character_forge: dropped invalid character %r: %s", entry.get("name"), exc)
    return out


def _enforce_budget(
    bibles: List[CharacterBibleV2],
    budget: Dict[str, Any],
) -> List[CharacterBibleV2]:
    """Clamp the cast to the budget by tier.

    If the LLM returned too many of a tier, the excess are demoted (main
    → recurring → side) or dropped (side over the ceiling). If too few,
    we accept the under-cap; the Architect / Chapter Planner can request
    more later via forge_additional_characters().
    """
    main_min, main_max = budget["main"]
    rec_min, rec_max = budget["recurring"]
    side_min, side_max = budget["side"]

    by_tier: Dict[CharacterTier, List[CharacterBibleV2]] = {
        CharacterTier.MAIN: [],
        CharacterTier.RECURRING: [],
        CharacterTier.SIDE: [],
    }
    for c in bibles:
        by_tier[c.tier].append(c)

    # Demote main → recurring if over cap
    while len(by_tier[CharacterTier.MAIN]) > main_max:
        demoted = by_tier[CharacterTier.MAIN].pop()
        demoted.tier = CharacterTier.RECURRING
        by_tier[CharacterTier.RECURRING].append(demoted)
    while len(by_tier[CharacterTier.RECURRING]) > rec_max:
        demoted = by_tier[CharacterTier.RECURRING].pop()
        demoted.tier = CharacterTier.SIDE
        demoted.voice_fingerprint = None
        by_tier[CharacterTier.SIDE].append(demoted)
    while len(by_tier[CharacterTier.SIDE]) > side_max:
        by_tier[CharacterTier.SIDE].pop()

    flat = (
        by_tier[CharacterTier.MAIN]
        + by_tier[CharacterTier.RECURRING]
        + by_tier[CharacterTier.SIDE]
    )
    return flat
