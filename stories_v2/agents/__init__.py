"""v2 agents (async LLM-driven functions).

Each agent is a thin wrapper around ``providers.get_router().chat_json``
with a typed input/output and a role-specific prompt. Agents do NOT
manage their own state — they take inputs, return outputs, and the
``engine/`` orchestrator persists them.

Phase 2 agents:
    profiler_v2      — builds ProfileV2 from bio + samples
    world_builder    — builds WorldBibleV2 from premise + genre
    character_forge  — builds cast of CharacterBibleV2 respecting budget
    paste_parser     — parses freeform notes into ParsedSurveyDraft

Phase 5+ agents (architect, chapter_planner, scene_writer, critics,
editor, continuity_v2, publisher_v2) land in their own files.
"""

from .architect import build_arc_plan, find_act_for_chapter
from .chapter_planner import plan_chapter
from .character_forge import forge_cast
from .editor import edit_scene, should_rerun_critics
from .paste_parser import parse_pasted_notes
from .profiler_v2 import build_profile
from .scene_writer import draft_scene
from .world_builder import build_world_bible

__all__ = [
    "build_arc_plan",
    "find_act_for_chapter",
    "plan_chapter",
    "forge_cast",
    "edit_scene",
    "should_rerun_critics",
    "parse_pasted_notes",
    "build_profile",
    "draft_scene",
    "build_world_bible",
]
