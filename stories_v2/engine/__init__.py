"""Engine orchestration for v2.

Three runners stack:
    orchestrator.run_story()       — top-level: survey → architect → chapter loop → publish
    chapter_runner.run_chapter()   — plan → run scenes → commit → continuity refresh
    scene_runner.run_scene()       — draft → critic panel → editor → humanise → commit

State lives in Mongo. Each runner reads its inputs and writes its
outputs there. The orchestrator emits SSE events at every milestone.
"""

from .orchestrator import launch_story_run, run_story_async
from .regenerate import (
    apply_manual_edit,
    cascade_regenerate_from,
    regenerate_single_scene,
)

__all__ = [
    "launch_story_run",
    "run_story_async",
    "apply_manual_edit",
    "cascade_regenerate_from",
    "regenerate_single_scene",
]
