"""Profiler v2 — generates a structured ProfileV2 from biographical inputs."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..prompts_v2.profiler import SYSTEM, build_user_prompt
from ..providers import Router, get_router
from ..schemas_v2 import EmotionalDefaults, LexicalFingerprint, ProfileV2Input

logger = logging.getLogger(__name__)


async def build_profile(
    inputs: ProfileV2Input,
    *,
    router: Optional[Router] = None,
) -> Dict[str, Any]:
    """Run the Profiler v2 LLM call.

    Returns a dict shaped like the Profiler JSON contract:
        lexical_fingerprint, emotional_defaults, preferred_phrases,
        banned_phrases, few_shot_samples, expertise_tags

    The caller wraps this in a full ProfileV2 envelope (profile_id, name,
    inputs, timestamps) and persists.
    """
    router = router or get_router()
    user_prompt = build_user_prompt(inputs.model_dump())

    try:
        result = await router.chat_json(
            role="profiler",
            system=SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=4096,
            temperature=0.6,
        )
    except Exception as exc:
        logger.warning("profiler_v2: LLM call failed (%s) — falling back to defaults", exc)
        return _default_profile_payload(inputs)

    # Defensive: ensure required keys exist with safe defaults.
    return {
        "lexical_fingerprint": _safe_lexical(result.get("lexical_fingerprint")),
        "emotional_defaults": _safe_emotional(result.get("emotional_defaults")),
        "preferred_phrases": list(result.get("preferred_phrases") or []),
        "banned_phrases": list(result.get("banned_phrases") or []),
        "few_shot_samples": list(result.get("few_shot_samples") or []),
        "expertise_tags": list(result.get("expertise_tags") or inputs.expertise_tags or []),
    }


def _safe_lexical(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return LexicalFingerprint().model_dump()
    try:
        return LexicalFingerprint.model_validate(raw).model_dump()
    except Exception:
        return LexicalFingerprint().model_dump()


def _safe_emotional(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return EmotionalDefaults().model_dump()
    try:
        return EmotionalDefaults.model_validate(raw).model_dump()
    except Exception:
        return EmotionalDefaults().model_dump()


def _default_profile_payload(inputs: ProfileV2Input) -> Dict[str, Any]:
    """Fallback when no provider is reachable.  Lets development continue
    offline; production should always have at least one provider configured."""
    return {
        "lexical_fingerprint": LexicalFingerprint().model_dump(),
        "emotional_defaults": EmotionalDefaults().model_dump(),
        "preferred_phrases": [],
        "banned_phrases": [],
        "few_shot_samples": [],
        "expertise_tags": list(inputs.expertise_tags or []),
    }
