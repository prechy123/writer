"""Working memory — last 2 chapters verbatim + current beat plan.

Always present in the Scene Writer prompt; never goes through retrieval.
Trimmed to a token budget by taking the chapter's head (first ~600
words) and tail (last ~600 words) — middle is summarised via the
running summary or skipped, because that's exactly what episodic
memory exists to surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .. import mongo


@dataclass
class WorkingMemory:
    """Always-injected slice for the Scene Writer."""

    last_chapter_excerpts: List[Dict[str, Any]] = field(default_factory=list)
    current_chapter_beat_plan: Optional[Dict[str, Any]] = None
    previous_scene_ending: Optional[str] = None


def _trim_chapter_text(text: str, *, head_words: int = 400, tail_words: int = 400) -> Dict[str, str]:
    """Return ``{head, tail}`` slices of a long chapter text."""
    words = text.split()
    if len(words) <= head_words + tail_words + 80:
        return {"head": text, "tail": ""}
    head = " ".join(words[:head_words])
    tail = " ".join(words[-tail_words:])
    return {"head": head, "tail": tail}


def assemble_working_memory(
    *,
    story_id: str,
    current_chapter_idx: int,
    current_scene_idx: int,
    recent_window: int = 2,
) -> WorkingMemory:
    """Pull the last ``recent_window`` committed chapters + the current
    chapter's beat plan + the most recent scene's closing prose.

    Chapters are loaded from ``scene_drafts_v2`` (per-scene rows) and
    stitched on the fly.
    """
    out = WorkingMemory()

    # Current chapter beat plan
    beat = mongo.col(mongo.COL_BEATS).find_one(
        {"story_id": story_id, "chapter_idx": current_chapter_idx}
    )
    if beat:
        out.current_chapter_beat_plan = {
            k: v
            for k, v in beat.items()
            if k not in {"_id", "embedding"}
        }

    # Previous scene's closing prose (last ~120 words)
    prev_scene_idx = current_scene_idx - 1
    if prev_scene_idx >= 0:
        prev = mongo.col(mongo.COL_SCENES).find_one(
            {
                "story_id": story_id,
                "chapter_idx": current_chapter_idx,
                "scene_idx": prev_scene_idx,
            },
            projection={"final_prose": 1},
        )
        if prev and prev.get("final_prose"):
            words = prev["final_prose"].split()
            out.previous_scene_ending = " ".join(words[-120:])
    elif current_chapter_idx > 0:
        last_scenes = list(
            mongo.col(mongo.COL_SCENES)
            .find(
                {"story_id": story_id, "chapter_idx": current_chapter_idx - 1},
                projection={"scene_idx": 1, "final_prose": 1},
            )
            .sort("scene_idx", -1)
            .limit(1)
        )
        if last_scenes and last_scenes[0].get("final_prose"):
            words = last_scenes[0]["final_prose"].split()
            out.previous_scene_ending = " ".join(words[-120:])

    # Last N chapters as head+tail excerpts (skip the current chapter)
    start_ch = max(0, current_chapter_idx - recent_window)
    for ch_idx in range(start_ch, current_chapter_idx):
        scenes = list(
            mongo.col(mongo.COL_SCENES)
            .find(
                {"story_id": story_id, "chapter_idx": ch_idx},
                projection={"scene_idx": 1, "final_prose": 1, "summary": 1},
            )
            .sort("scene_idx", 1)
        )
        if not scenes:
            continue
        chapter_text = "\n\n".join(s.get("final_prose") or "" for s in scenes)
        trimmed = _trim_chapter_text(chapter_text)
        out.last_chapter_excerpts.append(
            {
                "chapter_idx": ch_idx,
                "head": trimmed["head"],
                "tail": trimmed["tail"],
                "scene_count": len(scenes),
            }
        )

    return out
