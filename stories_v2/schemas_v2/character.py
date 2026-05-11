"""Character bible v2 schemas."""

from __future__ import annotations

import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import CharacterTier, PlutchikVector, SceneEmotionAxes
from .voice import VoiceFingerprint


class CharacterMoodSnapshot(BaseModel):
    """One snapshot in a character's evolving emotional history."""

    model_config = ConfigDict(extra="forbid")

    chapter_idx: int
    scene_idx: int
    plutchik: PlutchikVector = Field(default_factory=PlutchikVector)
    axes: SceneEmotionAxes = Field(default_factory=SceneEmotionAxes)
    last_event_summary: str = ""
    recorded_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class CharacterRelationship(BaseModel):
    """Edge in the cast relationship graph."""

    model_config = ConfigDict(extra="forbid")

    other_character_id: str
    other_name: Optional[str] = None
    nature: str = Field(
        default="acquaintance",
        description="ally | rival | mentor | family | romantic | enemy | acquaintance",
    )
    tension: int = Field(ge=1, le=10, default=5)
    history: str = ""


class CharacterBibleV2(BaseModel):
    """Per-character bible.

    Side-tier characters store a stub: name, role, one trait, one sample
    line; the rest is left empty. Main + recurring get the full payload.
    """

    model_config = ConfigDict(extra="allow")

    character_id: str
    story_id: str
    name: str = Field(min_length=1, max_length=120)
    tier: CharacterTier = CharacterTier.RECURRING
    role: str = Field(default="", max_length=200)
    short_description: str = Field(default="", max_length=600)
    portrait_blurb: str = Field(
        default="",
        max_length=600,
        description="1–2 sentence visual + vibe blurb shown to readers on the story page.",
    )

    # Deep-only fields (empty for side tier)
    age: Optional[str] = None
    pronouns: Optional[str] = None
    background: str = ""
    motivations: List[str] = Field(default_factory=list)
    fears: List[str] = Field(default_factory=list)
    secrets: List[str] = Field(default_factory=list)
    arc: str = ""
    education_access: str = ""
    resources_or_limitations: str = ""
    knowledge_sources: List[str] = Field(default_factory=list)
    relationships: List[CharacterRelationship] = Field(default_factory=list)

    # Voice + mood — main + recurring only
    voice_fingerprint: Optional[VoiceFingerprint] = None
    mood_state_history: List[CharacterMoodSnapshot] = Field(default_factory=list)

    # Tropes / tags
    webnovel_role_hook: str = ""
    progression_function: str = ""

    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
