"""World bible v2 schemas."""

from __future__ import annotations

import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class WorldFaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str = Field(default="", description="protagonist's allies / antagonists / rival sect / state / etc.")
    goals: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)
    notable_members: List[str] = Field(default_factory=list)
    relationship_to_protagonist: str = ""


class WorldRule(BaseModel):
    """A scoped worldbuilding rule.

    Mirrors v1's WorldRule but adds an introduced-as-context flag so the
    Architect can distinguish rules that always apply from ones tied to a
    specific arc.
    """

    model_config = ConfigDict(extra="forbid")

    scope: str = Field(description="e.g. 'Imperial capital', 'village', 'global', 'cultivation realm'")
    rule: str
    consequence_if_broken: str = ""


class MagicOrSystem(BaseModel):
    """Power-system spec (used heavily by progression web novels)."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(default="none", description="none | cultivation | litrpg | classical_magic | psionics | tech | other")
    description: str = ""
    progression_path: List[str] = Field(
        default_factory=list,
        description="Ordered tiers / ranks / realms / levels readers can track.",
    )
    cost_or_drawback: str = ""
    hard_limits: List[str] = Field(default_factory=list)


class WorldBibleV2(BaseModel):
    """Story-scoped worldbuilding bible."""

    model_config = ConfigDict(extra="allow")

    story_id: str
    setting: str = Field(default="", description="One paragraph anchor: time, place, scale, mood.")
    time_period: str = ""
    technology_level: str = ""
    social_structure: str = ""
    geography: str = ""
    languages: List[str] = Field(default_factory=list)
    factions: List[WorldFaction] = Field(default_factory=list)
    magic_or_system: MagicOrSystem = Field(default_factory=MagicOrSystem)
    rules: List[WorldRule] = Field(default_factory=list)
    banned_anachronisms: List[str] = Field(
        default_factory=list,
        description="Concepts/tech the writer must never reference (e.g. 'smartphone' in a medieval setting).",
    )
    must_have_vibes: List[str] = Field(
        default_factory=list,
        description="Atmospheric anchors: 'lantern-lit alleys', 'salt wind', 'incense haze'.",
    )
    notes: Optional[str] = None

    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
