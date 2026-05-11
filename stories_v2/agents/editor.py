"""Editor agent — single coherent rewrite using all critic findings."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..prompts_v2.editor import SYSTEM, build_user_prompt
from ..providers import Router, get_router
from ..schemas_v2 import CriticReport, SceneBeat

logger = logging.getLogger(__name__)


async def edit_scene(
    *,
    draft: str,
    scene_beat: SceneBeat,
    critic_reports: List[CriticReport],
    router: Optional[Router] = None,
) -> str:
    """Return revised prose. On LLM failure, returns the original draft."""
    router = router or get_router()

    overall_score = _overall_score(critic_reports)
    user_prompt = build_user_prompt(
        scene_beat_compact=_compact_scene_beat(scene_beat),
        draft_prose=draft,
        critic_reports=[r.model_dump() for r in critic_reports],
        overall_score=overall_score,
    )

    try:
        revised = await router.chat_text(
            role="editor",
            system=SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=_max_tokens_for(scene_beat.target_words),
            temperature=0.55,
            timeout=180.0,
        )
    except Exception as exc:
        logger.warning("editor: rewrite failed (%s); keeping prior draft", exc)
        return draft

    revised = (revised or "").strip()
    if not revised:
        return draft
    return revised


def should_rerun_critics(reports: List[CriticReport]) -> bool:
    """Decide whether the new draft warrants another critic pass.

    The scene_runner caps cycles regardless, but this helper lets the
    orchestrator skip a redundant pass when the editor already addressed
    everything.
    """
    if not reports:
        return False
    return any(r.score < 0.75 or any(f.severity == "error" for f in r.findings) for r in reports)


def _overall_score(reports: List[CriticReport]) -> float:
    if not reports:
        return 1.0
    # Worst-of-N — the Editor should focus on the harshest critique.
    return round(min(r.score for r in reports), 3)


def _max_tokens_for(target_words: int) -> int:
    target_words = max(target_words or 600, 300)
    return min(8000, int(target_words * 1.6 * 1.3))


def _compact_scene_beat(beat: SceneBeat) -> str:
    import json
    return json.dumps(beat.model_dump(), indent=2, default=str)
