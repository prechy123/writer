"""Continuity v2 — post-scene state refresh.

Called once per scene after the prose commits. Produces:
  - Updated character mood snapshots (added to mood_state_history)
  - Updated story-level continuity_ledger (open threads, plot seeds, world state)
  - Scene summary + key dialogue for episodic memory
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional

from ..emotion import apply_scene_event, plutchik_to_valence_arousal
from ..emotion.plutchik import from_mapping as plutchik_from_mapping
from ..prompts_v2.continuity import SYSTEM, build_user_prompt
from ..providers import Router, get_router
from ..schemas_v2 import (
    CharacterMoodSnapshot,
    PlutchikVector,
    SceneBeat,
)

logger = logging.getLogger(__name__)


async def refresh_continuity(
    *,
    story_id: str,
    chapter_idx: int,
    scene_idx: int,
    scene_beat: SceneBeat,
    final_prose: str,
    cast: List[Dict[str, Any]],
    arc_seeds: Optional[List[Dict[str, Any]]] = None,
    current_ledger: Optional[Dict[str, Any]] = None,
    router: Optional[Router] = None,
) -> Dict[str, Any]:
    """Return the raw continuity dict from the LLM, validated/cleaned.

    Caller persists it to Mongo + applies the mood snapshots to character
    bibles.
    """
    router = router or get_router()
    user_prompt = build_user_prompt(
        scene_beat=scene_beat.model_dump(),
        final_prose=final_prose,
        cast=cast,
        arc_seeds=arc_seeds,
        current_ledger=current_ledger,
    )

    try:
        raw = await router.chat_json(
            role="continuity",
            system=SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=4000,
            temperature=0.3,
        )
    except Exception as exc:
        logger.warning("continuity_v2: LLM call failed (%s); emitting empty refresh", exc)
        raw = {}

    if not isinstance(raw, dict):
        raw = {}

    # Defensive cleanup
    return {
        "scene_summary": str(raw.get("scene_summary") or "").strip(),
        "key_dialogue": [str(s).strip() for s in (raw.get("key_dialogue") or []) if s],
        "character_mood_deltas": _clean_mood_deltas(raw.get("character_mood_deltas") or [], cast),
        "protagonist_emotion_end": _safe_plutchik(raw.get("protagonist_emotion_end")),
        "reader_emotion_end": _safe_plutchik(raw.get("reader_emotion_end")),
        "world_state_changes": [str(s) for s in (raw.get("world_state_changes") or []) if s],
        "plot_seed_events": _clean_plot_seed_events(raw.get("plot_seed_events") or []),
        "open_threads_added": [str(s) for s in (raw.get("open_threads_added") or []) if s],
        "open_threads_closed": [str(s) for s in (raw.get("open_threads_closed") or []) if s],
        "unresolved_cliffhangers": [str(s) for s in (raw.get("unresolved_cliffhangers") or []) if s],
    }


def apply_mood_to_character(
    *,
    character_bible: Dict[str, Any],
    delta: Dict[str, float],
    chapter_idx: int,
    scene_idx: int,
    event_summary: str = "",
) -> CharacterMoodSnapshot:
    """Apply a Plutchik delta to a character's most recent mood snapshot."""
    history = character_bible.get("mood_state_history") or []
    if history:
        last = history[-1]
        try:
            prior = CharacterMoodSnapshot.model_validate(last)
        except Exception:
            prior = CharacterMoodSnapshot(
                chapter_idx=chapter_idx, scene_idx=scene_idx,
                plutchik=PlutchikVector(),
            )
    else:
        prior = CharacterMoodSnapshot(
            chapter_idx=chapter_idx, scene_idx=scene_idx,
            plutchik=PlutchikVector(trust=0.3, anticipation=0.3),
        )

    return apply_scene_event(
        prior,
        delta=delta,
        chapter_idx=chapter_idx,
        scene_idx=scene_idx,
        event_summary=event_summary,
        apply_decay=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_plutchik(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return PlutchikVector().model_dump()
    try:
        return plutchik_from_mapping(raw).model_dump()
    except Exception:
        return PlutchikVector().model_dump()


def _clean_mood_deltas(raw: List[Any], cast: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid_ids = {c.get("character_id") for c in cast if c.get("character_id")}
    out: List[Dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        cid = entry.get("character_id")
        if cid not in valid_ids:
            continue
        delta = entry.get("plutchik_delta") or {}
        clean_delta = {
            k: max(-0.5, min(0.5, float(v)))
            for k, v in delta.items()
            if k in ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")
            and v is not None
        }
        out.append({
            "character_id": cid,
            "name": entry.get("name") or "",
            "plutchik_delta": clean_delta,
            "last_event_summary": str(entry.get("last_event_summary") or "").strip(),
        })
    return out


def _clean_plot_seed_events(raw: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("seed_id")
        action = entry.get("action")
        if not sid or action not in ("developed", "paid_off", "planted"):
            continue
        out.append({"seed_id": sid, "action": action, "ts": datetime.datetime.utcnow().isoformat()})
    return out
