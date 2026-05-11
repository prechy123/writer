"""CorpusEntry schema."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CorpusEntry(BaseModel):
    """One exemplar passage.

    Used as a few-shot anchor in the Scene Writer prompt. Keep passages
    short (200-500 words ideal) — past that the LLM starts mimicking
    surface structure too literally.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    source: str = Field(
        description="'public_domain:<title> by <author>' or 'original_technique_example'.",
    )
    techniques: List[str] = Field(
        default_factory=list,
        description="Craft techniques the passage illustrates: 'sensory_anchoring', 'interruption', 'show_dont_tell', 'sentence_variety', 'subtext', 'mid_paragraph_tonal_shift', etc.",
    )
    genres: List[str] = Field(
        default_factory=list,
        description="'cultivation', 'litrpg', 'isekai', 'light_novel', 'romance', 'literary', 'generic'.",
    )
    pov: Optional[str] = Field(default=None, description="first | third_limited | third_omniscient")
    pacing: Optional[str] = Field(default=None, description="slow_burn | balanced | breakneck")
    emotion_tags: List[str] = Field(
        default_factory=list,
        description="Plutchik axes the passage illustrates: 'fear', 'anticipation', 'sadness', etc.",
    )
    style_register: Optional[str] = Field(default=None, description="casual | formal | archaic | street | ...")
    text: str
    notes: Optional[str] = None
