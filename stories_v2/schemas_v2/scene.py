"""Scene draft + critic finding schemas."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import PlutchikVector


Severity = Literal["info", "warn", "error"]
CriticName = Literal["voice", "emotion", "show_dont_tell", "ai_detect", "pacing"]


class CriticFinding(BaseModel):
    """One structured remark from a critic.

    Editor takes the union of all findings and synthesises a rewrite.
    """

    model_config = ConfigDict(extra="allow")

    critic: CriticName
    severity: Severity
    field: str = Field(default="", description="Which axis / phrase / property triggered the finding.")
    expected: str = ""
    observed: str = ""
    span: Optional[str] = Field(default=None, description="A literal substring of the prose, if applicable.")
    suggestion: Optional[str] = None
    rationale: Optional[str] = None


class CriticReport(BaseModel):
    """One critic's full output."""

    model_config = ConfigDict(extra="allow")

    critic: CriticName
    score: float = Field(ge=0.0, le=1.0, default=1.0)
    on_target: bool = True
    findings: List[CriticFinding] = Field(default_factory=list)
    notes: Optional[str] = None


class HumanisationReport(BaseModel):
    """Phase 7 deterministic-pipeline output."""

    model_config = ConfigDict(extra="allow")

    em_dash_replacements: int = 0
    banned_phrase_strikes: int = 0
    contractions_injected: int = 0
    fragments_injected: int = 0
    sentences_split: int = 0
    sentences_merged: int = 0
    burstiness_before: float = 0.0
    burstiness_after: float = 0.0
    detector_score_before: Optional[float] = None
    detector_score_after: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class SceneDraft(BaseModel):
    """A persisted scene record.

    Stored in ``scene_drafts_v2`` keyed by (story_id, chapter_idx,
    scene_idx).  ``final_prose`` is what readers see; ``draft_history``
    keeps prior versions for diffing / rollback.
    """

    model_config = ConfigDict(extra="allow")

    story_id: str
    chapter_idx: int = Field(ge=0)
    scene_idx: int = Field(ge=0)

    draft_prose: str = ""
    final_prose: str = ""
    draft_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="[{prose, source: writer|editor|user_edit|humaniser, recorded_at}]",
    )

    critic_reports: List[CriticReport] = Field(default_factory=list)
    humanisation_report: Optional[HumanisationReport] = None

    summary: str = Field(default="", description="One-paragraph summary used by episodic memory.")
    key_dialogue: List[str] = Field(default_factory=list)
    mood_snapshot: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None

    protagonist_emotion_after: Optional[PlutchikVector] = None
    reader_emotion_after: Optional[PlutchikVector] = None
    emotion_target_score: Optional[Dict[str, Any]] = None

    word_count: int = 0
    cycle_count: int = Field(ge=0, default=0)
    status: Literal["drafting", "critiquing", "editing", "humanising", "committed", "failed"] = "drafting"

    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    committed_at: Optional[datetime.datetime] = None
