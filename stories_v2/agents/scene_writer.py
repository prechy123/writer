"""Scene Writer agent — the prose-producing call.

Drafts ONE scene given a SceneBeat + memory + voice + corpus. Returns
the raw draft prose. Critic + editor + humaniser passes are separate
agents wired in the scene_runner orchestrator (Phase 8).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..corpus import load_index
from ..memory import MemoryContext
from ..prompts_v2.scene_writer import SYSTEM, build_user_prompt
from ..providers import Router, get_router
from ..schemas_v2 import CharacterBibleV2, SceneBeat
from ..voice import build_scene_few_shot

logger = logging.getLogger(__name__)


async def draft_scene(
    *,
    scene_beat: SceneBeat,
    memory: MemoryContext,
    present_characters: List[CharacterBibleV2],
    author_profile_hint: Optional[Dict[str, Any]] = None,
    router: Optional[Router] = None,
    target_words: Optional[int] = None,
) -> str:
    """Produce a first-pass scene draft. Returns prose only."""
    router = router or get_router()

    voice_few_shot = build_scene_few_shot(present_characters)
    corpus_exemplars = _resolve_exemplars(scene_beat.retrieved_exemplar_ids)

    working = _working_payload(memory)
    semantic_payload = {
        "present_characters": [c.model_dump() for c in present_characters],
        "world_bible": memory.world_bible,
        "relevant_world_rules": memory.relevant_world_rules,
        "active_threads": memory.active_threads,
    }

    user_prompt = build_user_prompt(
        scene_beat=scene_beat.model_dump(),
        voice_few_shot_block=voice_few_shot,
        corpus_exemplars=corpus_exemplars,
        working_memory=working,
        semantic_context=semantic_payload,
        episodic_excerpts=memory.episodic_scenes or [],
        author_profile_hint=author_profile_hint,
    )

    max_tokens = _scene_max_tokens(target_words or scene_beat.target_words)

    try:
        prose = await router.chat_text(
            role="scene_writer",
            system=SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=0.85,
            timeout=240.0,
        )
    except Exception as exc:
        logger.exception("scene_writer: LLM call failed")
        raise

    return (prose or "").strip()


def _resolve_exemplars(exemplar_ids: List[str]) -> List[Dict[str, Any]]:
    if not exemplar_ids:
        return []
    by_id = {e.id: e for e in load_index()}
    out: List[Dict[str, Any]] = []
    for eid in exemplar_ids:
        if eid in by_id:
            out.append(by_id[eid].model_dump())
    return out


def _working_payload(memory: MemoryContext) -> Dict[str, Any]:
    return {
        "previous_scene_ending": memory.working.previous_scene_ending,
        "last_chapter_excerpts": memory.working.last_chapter_excerpts,
        "current_chapter_beat_plan": memory.working.current_chapter_beat_plan,
    }


def _scene_max_tokens(target_words: int) -> int:
    """Token budget ~ 1.6 tokens/word for English, + 30% headroom."""
    target_words = max(target_words or 600, 300)
    return min(8000, int(target_words * 1.6 * 1.3))
