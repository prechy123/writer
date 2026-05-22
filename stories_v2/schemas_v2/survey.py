"""Survey intake schemas (Quick / Deep / Paste-anything)."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .character import CharacterRelationship
from .common import CharacterTier
from .world import MagicOrSystem, WorldFaction, WorldRule


# ---------------------------------------------------------------------------
# Quick wizard (5 fields, ~5 min)
# ---------------------------------------------------------------------------

class QuickCharacter(BaseModel):
    """Minimal character input for the Quick wizard."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    role: str = Field(default="protagonist", max_length=120)
    trait: str = Field(default="", max_length=300, description="One vivid trait.")
    sample_line: Optional[str] = Field(default=None, max_length=400)


class QuickSurvey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=300)
    premise: str = Field(min_length=10, max_length=4000)
    num_chapters: int = Field(ge=1, le=200)
    initial_chapters: Optional[int] = Field(default=None, ge=1, le=200)
    genres: List[str] = Field(
        default_factory=list,
        description="e.g. ['progression_fantasy','cultivation'] or ['light_novel','isekai']",
    )
    tone: List[str] = Field(
        default_factory=list,
        description="multi-select: gritty | hopeful | dry_funny | tragic | cathartic | wholesome | dark",
    )
    characters: List[QuickCharacter] = Field(default_factory=list, max_length=8)
    pov: Literal["first", "third_limited", "third_omniscient", "multi"] = "third_limited"
    tense: Literal["past", "present"] = "past"
    target_chapter_words: int = Field(ge=600, le=8000, default=2200)
    profile_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Deep wizard (additive: full character + world bibles)
# ---------------------------------------------------------------------------

class DeepCharacter(BaseModel):
    """Full character input for the Deep wizard."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    tier: CharacterTier = CharacterTier.MAIN
    role: str = Field(default="", max_length=200)
    pronouns: Optional[str] = None
    age: Optional[str] = None
    background: str = Field(default="", max_length=4000)
    motivations: List[str] = Field(default_factory=list)
    fears: List[str] = Field(default_factory=list)
    secrets: List[str] = Field(default_factory=list)
    arc: str = Field(default="", max_length=2000)
    education_access: str = Field(default="", max_length=1000)
    resources_or_limitations: str = Field(default="", max_length=1000)
    knowledge_sources: List[str] = Field(default_factory=list)
    relationships: List[CharacterRelationship] = Field(default_factory=list)

    # Voice — user can leave blank and Character Forge will infer
    speech_traits: str = Field(default="", max_length=2000)
    preferred_phrases: List[str] = Field(default_factory=list)
    banned_phrases: List[str] = Field(default_factory=list)
    sample_lines: List[str] = Field(default_factory=list, max_length=12)

    webnovel_role_hook: str = ""
    progression_function: str = ""


class DeepWorld(BaseModel):
    model_config = ConfigDict(extra="forbid")

    setting: str = ""
    time_period: str = ""
    technology_level: str = ""
    social_structure: str = ""
    geography: str = ""
    languages: List[str] = Field(default_factory=list)
    factions: List[WorldFaction] = Field(default_factory=list)
    magic_or_system: MagicOrSystem = Field(default_factory=MagicOrSystem)
    rules: List[WorldRule] = Field(default_factory=list)
    banned_anachronisms: List[str] = Field(default_factory=list)
    must_have_vibes: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class DeepArcPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    must_include_tropes: List[str] = Field(default_factory=list)
    must_avoid_tropes: List[str] = Field(default_factory=list)
    progression_milestones: List[str] = Field(
        default_factory=list,
        description="Visible wins (power tier, rank, romance step) you want delivered, in order.",
    )
    plot_seeds_to_plant: List[str] = Field(default_factory=list)
    reader_emotional_journey: str = ""
    cliffhanger_intensity: Literal["low", "medium", "high"] = "high"
    pacing_speed: Literal["slow_burn", "balanced", "breakneck"] = "balanced"
    romance_temperature: Literal["none", "subtext", "warm", "spicy"] = "subtext"
    action_density: Literal["light", "balanced", "heavy"] = "balanced"


class DeepStyleAnchors(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_authors: List[str] = Field(default_factory=list)
    reference_books: List[str] = Field(default_factory=list)
    pasted_sample_passages: List[str] = Field(default_factory=list)


class DeepSurvey(BaseModel):
    """Quick wizard fields + full bibles + arc + style anchors."""

    model_config = ConfigDict(extra="forbid")

    quick: QuickSurvey
    characters: List[DeepCharacter] = Field(default_factory=list, max_length=120)
    world: DeepWorld = Field(default_factory=DeepWorld)
    arc_preferences: DeepArcPreferences = Field(default_factory=DeepArcPreferences)
    style_anchors: DeepStyleAnchors = Field(default_factory=DeepStyleAnchors)


# ---------------------------------------------------------------------------
# Paste-anything intake
# ---------------------------------------------------------------------------

class PastedNotes(BaseModel):
    """User paste — anything from a hand-written world bible to draft prose."""

    model_config = ConfigDict(extra="forbid")

    raw_text: str = Field(min_length=20, max_length=200_000)
    hint: Optional[str] = Field(
        default=None,
        description="Optional hint to the parser: e.g. 'this is a character bible', 'this is a sample chapter'.",
    )


class ParsedSurveyDraft(BaseModel):
    """Parser output — used to pre-fill the Deep wizard for user review."""

    model_config = ConfigDict(extra="allow")

    quick: Optional[QuickSurvey] = None
    characters: List[DeepCharacter] = Field(default_factory=list)
    world: Optional[DeepWorld] = None
    arc_preferences: Optional[DeepArcPreferences] = None
    style_anchors: Optional[DeepStyleAnchors] = None
    notes: List[str] = Field(default_factory=list, description="Parser notes / caveats to show the user.")


# ---------------------------------------------------------------------------
# Import-past-chapters intake
# ---------------------------------------------------------------------------

class ImportedChapter(BaseModel):
    """One pasted prior chapter that becomes canon for the new story."""

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(default=None, max_length=300)
    text: str = Field(min_length=50, max_length=200_000)


class ImportSurvey(BaseModel):
    """Import-past-chapters intake: canon prose + continuation brief + profile choice.

    The pasted chapters become the story-so-far (chapters 1..N). The engine
    generates ``chapters_to_generate`` new chapters from N+1 onward. If
    ``end_story`` is True, the last generated chapter is written as a finale
    and the story is marked completed.
    """

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(default=None, max_length=300)
    chapters: List[ImportedChapter] = Field(min_length=1, max_length=100)
    description: str = Field(
        min_length=20,
        max_length=20_000,
        description="How the story should go from chapter N+1 onward.",
    )
    chapters_to_generate: int = Field(ge=1, le=200)
    end_story: bool = False

    profile_mode: Literal["select", "generate"]
    profile_id: Optional[str] = None          # when profile_mode == "select"
    new_profile_name: Optional[str] = Field(default=None, max_length=200)
    new_profile_bio: Optional[str] = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def _check_profile_mode_fields(self) -> "ImportSurvey":
        if self.profile_mode == "select":
            if not self.profile_id:
                raise ValueError("profile_id is required when profile_mode == 'select'.")
            if self.new_profile_name or self.new_profile_bio:
                raise ValueError(
                    "new_profile_name / new_profile_bio must be empty when profile_mode == 'select'."
                )
        else:  # profile_mode == "generate"
            if not (self.new_profile_name or "").strip():
                raise ValueError(
                    "new_profile_name is required when profile_mode == 'generate'."
                )
            if self.profile_id:
                raise ValueError(
                    "profile_id must be empty when profile_mode == 'generate'."
                )
        return self
