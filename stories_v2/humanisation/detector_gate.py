"""Optional external AI-detector gate.

Disabled unless ``settings.STORIES_V2_AI_DETECTOR_KEY`` is set. When
enabled, the orchestrator runs the final prose through the configured
detector (currently GPTZero-compatible JSON API) and stores the score
on the SceneDraft.

If the score exceeds the configured threshold (default 0.30 = 30%
likely-AI), the orchestrator can trigger a surgical re-edit pass. The
gate itself does not rewrite — that's the orchestrator's call.

Out of scope: we don't ship a vendor-specific implementation here.
The hook is in place; users with a paid detector key can register a
callback later.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# A pluggable callback. If set, it's called as `await callback(prose) -> float in [0, 1]`.
_detector_callback: Optional[Callable[[str], Awaitable[float]]] = None


def register_detector(cb: Callable[[str], Awaitable[float]]) -> None:
    """Register a custom detector callback (typically wired during app startup)."""
    global _detector_callback
    _detector_callback = cb


async def score(prose: str) -> Optional[float]:
    """Return a 0..1 AI-likelihood score, or None if no detector is wired."""
    if _detector_callback is None:
        return None
    try:
        result = await _detector_callback(prose)
        return float(max(0.0, min(1.0, result)))
    except Exception as exc:
        logger.warning("detector_gate: callback failed (%s)", exc)
        return None
