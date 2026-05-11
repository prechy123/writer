"""Emotion critic — purely LLM-driven (Plutchik inference from prose is
hard to do reliably with regex). The score_target_delivery helper is
used post-hoc by the orchestrator once Continuity v2 has inferred the
delivered vectors."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ...prompts_v2.critics import EMOTION_CRITIC_SYSTEM, build_critic_user_prompt
from ...providers import Router
from ...schemas_v2 import CriticReport
from ._common import call_critic_llm, compact_scene_beat, parse_critic_report


async def critique_emotion(
    *,
    draft: str,
    scene_beat: Dict[str, Any],
    router: Optional[Router] = None,
) -> CriticReport:
    user_prompt = build_critic_user_prompt(
        scene_beat_compact=compact_scene_beat(scene_beat),
        draft_prose=draft,
    )
    raw = await call_critic_llm(
        role="critic_emotion",
        system=EMOTION_CRITIC_SYSTEM,
        user_prompt=user_prompt,
        critic_name="emotion",
        router=router,
    )
    return parse_critic_report(raw, critic_name="emotion")
