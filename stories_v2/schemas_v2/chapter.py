"""ChapterPlanV2 + SceneBeat schemas — Kishōtenketsu per chapter."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import PlutchikVector


SensoryAxis = Literal["sight", "sound", "smell", "touch", "taste"]
InteriorityDensity = Literal["low", "medium", "high"]
KishoPhase = Literal["introduction", "development", "twist", "conclusion"]


class SceneBeat(BaseModel):
    """One scene's plan — what the Scene Writer takes as input."""

    model_config = ConfigDict(extra="allow")

    scene_idx: int = Field(ge=0)
    title: str = Field(default="", max_length=200)
    summary: str = Field(default="", max_length=1000)

    kisho_phase: KishoPhase = "development"
    pov_character_id: str
    pov_character_name: str = ""
    present_character_ids: List[str] = Field(default_factory=list)

    location: str = ""
    time_of_day: str = ""

    # Dwight Swain Scene structure (Goal / Conflict / Disaster)
    goal: str = Field(default="", description="What the POV character wants this scene")
    conflict: str = Field(default="", description="What gets in the way")
    disaster: str = Field(default="", description="How the scene ends (worse off, complicated, or escalated)")

    # Emotion + craft targets (Plutchik vectors set by the Chapter Planner)
    protagonist_start_emotion: PlutchikVector = Field(default_factory=PlutchikVector)
    protagonist_end_emotion: PlutchikVector = Field(default_factory=PlutchikVector)
    reader_start_emotion: PlutchikVector = Field(default_factory=PlutchikVector)
    reader_end_emotion: PlutchikVector = Field(default_factory=PlutchikVector)

    sensory_focus: List[SensoryAxis] = Field(default_factory=list)
    interiority_density: InteriorityDensity = "medium"
    techniques: List[str] = Field(
        default_factory=list,
        description="Craft techniques to apply (matches corpus exemplar tags).",
    )

    retrieved_exemplar_ids: List[str] = Field(default_factory=list)
    target_words: int = Field(ge=200, le=4000, default=600)

    notes: Optional[str] = None


class ChapterPlanV2(BaseModel):
    """Per-chapter plan: 3-6 SceneBeats following Kishōtenketsu."""

    model_config = ConfigDict(extra="allow")

    story_id: str
    chapter_idx: int = Field(ge=0)
    chapter_number: int = Field(ge=1)
    chapter_title: str = Field(default="", max_length=200)
    chapter_summary: str = Field(default="", max_length=2000)

    act_name: Literal["discovery", "escalation", "revelation", "catharsis"] = "discovery"
    chapter_position_in_act: float = Field(ge=0.0, le=1.0, default=0.0)

    opening_hook: str = Field(default="", description="The first-200-words hook")
    cliffhanger: str = Field(default="", description="Last-page anchor that pulls readers forward")
    progression_reward: str = Field(
        default="",
        description="Visible win to deliver in this chapter (linked to a milestone).",
    )

    scenes: List[SceneBeat] = Field(default_factory=list)

    target_chapter_words: int = Field(ge=600, le=8000, default=2200)
    target_word_floor: int = Field(ge=0, le=8000, default=1500)
    target_word_ceiling: int = Field(ge=600, le=8000, default=3200)

    # Cross-refs the Architect / Continuity track
    progression_milestone_id: Optional[str] = None
    plot_seed_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="[{seed_id, action: plant|develop|payoff}, ...]",
    )

    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
