"""Pydantic schemas shared across the v2 engine.

Submodules:
    common    — enums + lightweight primitives (status, tiers, events)
    voice     — LexicalFingerprint + VoiceFingerprint
    profile   — ProfileV2, EmotionalDefaults
    character — CharacterBibleV2, CharacterMoodSnapshot, CharacterRelationship
    world     — WorldBibleV2, WorldFaction, WorldRule, MagicOrSystem
    survey    — QuickSurvey, DeepSurvey, PastedNotes, ParsedSurveyDraft
    arc       — ArcPlan, ProgressionMilestone, PlotSeed (Phase 5)
    chapter   — ChapterPlanV2, SceneBeat (Phase 5)
    scene     — SceneDraft, SceneEmotionTarget, CriticFinding (Phase 5/6)
    manuscript— ManuscriptV2 (Phase 8)
"""

from .character import (
    CharacterBibleV2,
    CharacterMoodSnapshot,
    CharacterRelationship,
)
from .common import (
    CHARACTER_BUDGET_TIERS,
    CharacterTier,
    PlutchikVector,
    ProgressEvent,
    RunEvent,
    SceneEmotionAxes,
    StoryStatus,
    character_budget_for,
)
from .profile import EmotionalDefaults, ProfileV2, ProfileV2Input
from .survey import (
    DeepArcPreferences,
    DeepCharacter,
    DeepStyleAnchors,
    DeepSurvey,
    DeepWorld,
    ImportedChapter,
    ImportSurvey,
    ParsedSurveyDraft,
    PastedNotes,
    QuickCharacter,
    QuickSurvey,
)
from .arc import (
    ActName,
    ActPlan,
    ArcPlan,
    PlotSeed,
    ProgressionMilestone,
    Subplot,
)
from .chapter import (
    ChapterPlanV2,
    InteriorityDensity,
    KishoPhase,
    SceneBeat,
    SensoryAxis,
)
from .scene import (
    CriticFinding,
    CriticName,
    CriticReport,
    HumanisationReport,
    SceneDraft,
    Severity,
)
from .voice import LexicalFingerprint, VoiceFingerprint
from .world import MagicOrSystem, WorldBibleV2, WorldFaction, WorldRule

__all__ = [
    # common
    "CHARACTER_BUDGET_TIERS",
    "CharacterTier",
    "PlutchikVector",
    "ProgressEvent",
    "RunEvent",
    "SceneEmotionAxes",
    "StoryStatus",
    "character_budget_for",
    # voice
    "LexicalFingerprint",
    "VoiceFingerprint",
    # profile
    "EmotionalDefaults",
    "ProfileV2",
    "ProfileV2Input",
    # character
    "CharacterBibleV2",
    "CharacterMoodSnapshot",
    "CharacterRelationship",
    # world
    "MagicOrSystem",
    "WorldBibleV2",
    "WorldFaction",
    "WorldRule",
    # arc
    "ActName",
    "ActPlan",
    "ArcPlan",
    "PlotSeed",
    "ProgressionMilestone",
    "Subplot",
    # chapter
    "ChapterPlanV2",
    "InteriorityDensity",
    "KishoPhase",
    "SceneBeat",
    "SensoryAxis",
    # scene
    "CriticFinding",
    "CriticName",
    "CriticReport",
    "HumanisationReport",
    "SceneDraft",
    "Severity",
    # survey
    "DeepArcPreferences",
    "DeepCharacter",
    "DeepStyleAnchors",
    "DeepSurvey",
    "DeepWorld",
    "ImportedChapter",
    "ImportSurvey",
    "ParsedSurveyDraft",
    "PastedNotes",
    "QuickCharacter",
    "QuickSurvey",
]
