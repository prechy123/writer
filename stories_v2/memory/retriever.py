"""3-tier memory orchestrator.

Single entry point ``assemble_context(...)`` that returns a
``MemoryContext`` containing all three tiers. The Scene Writer takes
this and formats it into its prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import episodic, semantic
from .working import WorkingMemory, assemble_working_memory

logger = logging.getLogger(__name__)


@dataclass
class MemoryContext:
    """Everything the Scene Writer needs about the story so far."""

    working: WorkingMemory = field(default_factory=WorkingMemory)
    episodic_scenes: List[Dict[str, Any]] = field(default_factory=list)
    present_characters: List[Dict[str, Any]] = field(default_factory=list)
    world_bible: Optional[Dict[str, Any]] = None
    relevant_world_rules: List[Dict[str, Any]] = field(default_factory=list)
    active_threads: Dict[str, List[Any]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


async def assemble_context(
    *,
    story_id: str,
    current_chapter_idx: int,
    current_scene_idx: int,
    scene_beat: Dict[str, Any],
    present_character_ids: List[str],
    scene_location: Optional[str] = None,
    recent_window: int = 2,
    episodic_k: int = 5,
) -> MemoryContext:
    """Build the full memory context for one scene.

    Working memory is pulled synchronously (it's pure Mongo). Episodic
    memory requires an embed call so it's async.
    """
    ctx = MemoryContext()
    notes: List[str] = []

    # Working
    ctx.working = assemble_working_memory(
        story_id=story_id,
        current_chapter_idx=current_chapter_idx,
        current_scene_idx=current_scene_idx,
        recent_window=recent_window,
    )

    # Semantic
    ctx.world_bible = semantic.fetch_world_bible(story_id=story_id)
    ctx.present_characters = semantic.fetch_present_characters(
        story_id=story_id,
        character_ids=present_character_ids,
    )
    ctx.relevant_world_rules = semantic.relevant_world_rules(
        ctx.world_bible, scene_location=scene_location
    )
    ctx.active_threads = semantic.fetch_active_threads(story_id=story_id)

    # Episodic
    try:
        ctx.episodic_scenes = await episodic.retrieve(
            story_id=story_id,
            scene_beat=scene_beat,
            k=episodic_k,
            exclude_chapter_idx_geq=max(0, current_chapter_idx - recent_window),
        )
    except Exception as exc:
        logger.warning("memory: episodic retrieval failed (%s); proceeding without", exc)
        notes.append(f"episodic_retrieval_failed:{type(exc).__name__}")

    ctx.notes = notes
    return ctx
