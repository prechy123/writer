"""ArcPlan schemas — the macro 4-act web-novel structure."""

from __future__ import annotations

import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


ActName = Literal["discovery", "escalation", "revelation", "catharsis"]


class ProgressionMilestone(BaseModel):
    """A visible win the reader can track.

    Power tier, status rank, romance step, territory claim, system level —
    whatever the genre uses to mark forward motion. The Architect plants
    these in chapter order; the Chapter Planner is asked to deliver them
    on schedule.
    """

    model_config = ConfigDict(extra="forbid")

    milestone_id: str
    name: str
    category: Literal["power", "status", "romance", "wealth", "territory", "knowledge", "skill", "team"] = "power"
    description: str = ""
    target_chapter_idx: int = Field(ge=0)
    delivered: bool = False
    delivered_in_chapter_idx: Optional[int] = None


class PlotSeed(BaseModel):
    """A planted mystery / setup beat.

    The Architect lists what gets planted, in which chapter, with a
    target payoff. Continuity v2 marks them as delivered post-payoff.
    """

    model_config = ConfigDict(extra="forbid")

    seed_id: str
    title: str
    summary: str
    planted_chapter_idx: int = Field(ge=0)
    payoff_target_chapter_idx: Optional[int] = None
    status: Literal["planted", "developing", "paying_off", "resolved", "abandoned"] = "planted"
    notes: Optional[str] = None


class Subplot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subplot_id: str
    name: str
    type: Literal["romance", "rivalry", "mentorship", "betrayal", "mystery", "redemption", "team", "other"] = "other"
    summary: str
    involved_character_ids: List[str] = Field(default_factory=list)
    start_chapter_idx: int = Field(ge=0)
    end_chapter_idx: Optional[int] = None
    status: Literal["active", "dormant", "resolved"] = "active"


class ActPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ActName
    chapter_range: tuple[int, int] = Field(description="(start_idx_inclusive, end_idx_inclusive)")
    promise: str = Field(description="What the reader is promised will happen in this act.")
    key_beats: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class ArcPlan(BaseModel):
    """Macro arc for one batch / book."""

    model_config = ConfigDict(extra="allow")

    story_id: str
    arc_name: str = "Primary Arc"
    arc_theme: str = ""
    target_reader_journey: str = ""

    num_chapters: int = Field(ge=1)
    acts: List[ActPlan] = Field(default_factory=list)
    progression_milestones: List[ProgressionMilestone] = Field(default_factory=list)
    plot_seeds: List[PlotSeed] = Field(default_factory=list)
    subplots: List[Subplot] = Field(default_factory=list)

    # Web-novel discovery shape
    must_include_tropes: List[str] = Field(default_factory=list)
    must_avoid_tropes: List[str] = Field(default_factory=list)
    cliffhanger_intensity: Literal["low", "medium", "high"] = "high"
    pacing_speed: Literal["slow_burn", "balanced", "breakneck"] = "balanced"
    romance_temperature: Literal["none", "subtext", "warm", "spicy"] = "subtext"
    action_density: Literal["light", "balanced", "heavy"] = "balanced"

    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
