"""Critic agents — five specialists that judge a scene draft.

All critics share the same interface:
    async def critique(*, draft, scene_beat, **extra) -> CriticReport

They run in parallel via ``asyncio.gather`` from the scene_runner.
Heuristic critics never fail (they run pure Python). LLM critics fail
soft: on provider failure they return an empty CriticReport so the
Editor still gets something to work with.
"""

from .ai_detect import critique_ai_detect
from .emotion import critique_emotion
from .pacing import critique_pacing
from .show_dont_tell import critique_show_dont_tell
from .voice import critique_voice

__all__ = [
    "critique_voice",
    "critique_emotion",
    "critique_show_dont_tell",
    "critique_ai_detect",
    "critique_pacing",
]
