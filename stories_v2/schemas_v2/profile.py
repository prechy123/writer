"""Profile v2 schemas — reusable author voice across stories."""

from __future__ import annotations

import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .voice import LexicalFingerprint


class EmotionalDefaults(BaseModel):
    """How this author handles emotion when nothing else specifies."""

    model_config = ConfigDict(extra="forbid")

    default_valence: float = Field(ge=-1.0, le=1.0, default=0.0)
    default_arousal: float = Field(ge=0.0, le=1.0, default=0.5)
    vulnerability_handling: str = Field(
        default="indirect",
        description="direct | indirect | deflected_with_humour | through_action",
    )
    humor_type: str = Field(
        default="dry",
        description="dry | absurd | bleak | warm | self-deprecating | crude | observational | none",
    )
    interiority_density: str = Field(
        default="medium",
        description="low | medium | high — ratio of inner thoughts to action/dialogue",
    )


class ProfileV2Input(BaseModel):
    """User-supplied inputs preserved for re-generation / editing."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    bio_context: str = Field(default="", max_length=10_000)
    background: str = Field(default="", max_length=4_000)
    personality: str = Field(default="", max_length=4_000)
    communication_style: str = Field(default="", max_length=4_000)
    interests_and_values: str = Field(default="", max_length=4_000)
    quirks: str = Field(default="", max_length=4_000)
    additional_context: str = Field(default="", max_length=4_000)
    writing_samples: List[str] = Field(default_factory=list)
    expertise_tags: List[str] = Field(default_factory=list)


class ProfileV2(BaseModel):
    """Full v2 profile document.

    Replaces v1's three free-text blobs (author_profile / emotional_guidelines
    / expert_styles) with a structured fingerprint that the v2 writer
    actually knows how to enforce.
    """

    model_config = ConfigDict(extra="allow")

    profile_id: str
    name: str
    inputs: ProfileV2Input
    lexical_fingerprint: LexicalFingerprint = Field(default_factory=LexicalFingerprint)
    emotional_defaults: EmotionalDefaults = Field(default_factory=EmotionalDefaults)
    preferred_phrases: List[str] = Field(default_factory=list)
    banned_phrases: List[str] = Field(default_factory=list)
    few_shot_samples: List[str] = Field(
        default_factory=list,
        description="5–8 short passages (~150 words) that exemplify the author's voice.",
    )
    expertise_tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
