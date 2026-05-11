"""Lightweight schemas shared across the v2 engine.

Kept dependency-light so any module (mongo accessors, views, agents) can
import them without pulling in heavy provider clients.
"""

from __future__ import annotations

import datetime
import enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class StoryStatus(str, enum.Enum):
    """Lifecycle states for a v2 story document."""

    PENDING = "pending"
    ARCHITECTING = "architecting"
    PLANNING = "planning"
    WRITING = "writing"
    AWAITING_CONTINUE = "awaiting_continue"
    COMPLETED = "completed"
    FAILED = "failed"


class CharacterTier(str, enum.Enum):
    """Cast-size budget tiers tied to chapter count.

    Main characters are POV-eligible and carry full voice fingerprints +
    mood vectors. Recurring characters appear in >=3 chapters. Side
    characters are stubs (no fingerprint, no mood tracking).
    """

    MAIN = "main"
    RECURRING = "recurring"
    SIDE = "side"


class PlutchikVector(BaseModel):
    """Plutchik's 8 primary emotions, each on [0, 1].

    Used for per-character mood state and per-scene emotion targets.
    """

    model_config = ConfigDict(extra="forbid")

    joy: float = Field(ge=0.0, le=1.0, default=0.0)
    trust: float = Field(ge=0.0, le=1.0, default=0.0)
    fear: float = Field(ge=0.0, le=1.0, default=0.0)
    surprise: float = Field(ge=0.0, le=1.0, default=0.0)
    sadness: float = Field(ge=0.0, le=1.0, default=0.0)
    disgust: float = Field(ge=0.0, le=1.0, default=0.0)
    anger: float = Field(ge=0.0, le=1.0, default=0.0)
    anticipation: float = Field(ge=0.0, le=1.0, default=0.0)


class SceneEmotionAxes(BaseModel):
    """Auxiliary mood axes that don't fit Plutchik cleanly."""

    model_config = ConfigDict(extra="forbid")

    valence: float = Field(ge=-1.0, le=1.0, default=0.0)
    arousal: float = Field(ge=0.0, le=1.0, default=0.5)


class ProgressEvent(BaseModel):
    """A single progress-bar checkpoint.

    Persisted on the story document for poll-based UIs and also emitted
    as an SSE event for streaming clients.
    """

    model_config = ConfigDict(extra="allow")

    stage: str
    message: str = ""
    percent: int = Field(ge=0, le=100, default=0)
    current_chapter: Optional[int] = None
    completed_chapters: int = 0
    total_chapters: int = 0
    error: Optional[str] = None
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class RunEvent(BaseModel):
    """An SSE event in the run_events_v2 capped collection.

    seq is monotonic per story so reconnecting clients can request
    ``after_seq=N`` to replay missed events.
    """

    model_config = ConfigDict(extra="allow")

    story_id: str
    seq: int
    event_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


# Used by the character forge + architect to enforce reasonable cast sizes.
# (See plan §"Character budgeting".)
CHARACTER_BUDGET_TIERS: list[Dict[str, Any]] = [
    {"max_chapters": 3, "main": (1, 1), "recurring": (1, 2), "side": (0, 2)},
    {"max_chapters": 7, "main": (1, 2), "recurring": (2, 3), "side": (2, 5)},
    {"max_chapters": 15, "main": (2, 3), "recurring": (3, 5), "side": (5, 10)},
    {"max_chapters": 25, "main": (2, 4), "recurring": (5, 8), "side": (10, 18)},
    {"max_chapters": 40, "main": (3, 5), "recurring": (7, 12), "side": (15, 28)},
    {"max_chapters": 10_000, "main": (4, 6), "recurring": (10, 18), "side": (25, 60)},
]


def character_budget_for(num_chapters: int) -> Dict[str, Any]:
    """Return the cast-size budget bucket for a given chapter count."""
    for row in CHARACTER_BUDGET_TIERS:
        if num_chapters <= row["max_chapters"]:
            return {
                "num_chapters": num_chapters,
                "main": row["main"],
                "recurring": row["recurring"],
                "side": row["side"],
            }
    # Fallback (shouldn't hit — last row has effectively unbounded ceiling).
    last = CHARACTER_BUDGET_TIERS[-1]
    return {
        "num_chapters": num_chapters,
        "main": last["main"],
        "recurring": last["recurring"],
        "side": last["side"],
    }
