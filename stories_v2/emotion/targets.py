"""Scene emotion target schema + delivery scoring."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..schemas_v2 import PlutchikVector, SceneEmotionAxes
from . import plutchik
from .valence_arousal import plutchik_to_valence_arousal, valence_arousal_distance


class SceneEmotionTarget(BaseModel):
    """What the Chapter Planner sets, what the Emotion Critic measures."""

    model_config = ConfigDict(extra="forbid")

    pov_character_id: str
    pov_character_name: str
    protagonist_start: PlutchikVector = Field(default_factory=PlutchikVector)
    protagonist_end: PlutchikVector = Field(default_factory=PlutchikVector)
    reader_start: PlutchikVector = Field(default_factory=PlutchikVector)
    reader_end: PlutchikVector = Field(default_factory=PlutchikVector)
    sensory_focus: List[Literal["sight", "sound", "smell", "touch", "taste"]] = Field(
        default_factory=list,
        description="The senses the scene MUST anchor in.  Show-Don't-Tell critic checks.",
    )
    interiority_density: Literal["low", "medium", "high"] = "medium"
    notes: Optional[str] = None


class SceneTargetReport(BaseModel):
    """Emotion Critic output."""

    model_config = ConfigDict(extra="forbid")

    protagonist_distance: float
    reader_distance: float
    va_distance: float
    on_target: bool
    severity: Literal["info", "warn", "error"] = "info"
    note: str = ""


def build_scene_target(
    *,
    pov_character_id: str,
    pov_character_name: str,
    protagonist_current_mood: PlutchikVector,
    target_reader_vector: PlutchikVector,
    sensory_focus: Optional[List[str]] = None,
    interiority_density: str = "medium",
    notes: Optional[str] = None,
) -> SceneEmotionTarget:
    """Build a SceneEmotionTarget.

    The Chapter Planner is responsible for deciding what end-state the
    protagonist should reach. As a default we tilt slightly toward the
    target reader vector (the protagonist's emotional state is often
    the *reader's* lens) — but the Architect / Chapter Planner is free
    to override.
    """
    protagonist_end = plutchik.blend(protagonist_current_mood, target_reader_vector, weight_b=0.35)
    return SceneEmotionTarget(
        pov_character_id=pov_character_id,
        pov_character_name=pov_character_name,
        protagonist_start=protagonist_current_mood,
        protagonist_end=protagonist_end,
        reader_start=protagonist_current_mood,
        reader_end=target_reader_vector,
        sensory_focus=list(sensory_focus or []),  # type: ignore[arg-type]
        interiority_density=interiority_density,  # type: ignore[arg-type]
        notes=notes,
    )


def score_target_delivery(
    target: SceneEmotionTarget,
    *,
    delivered_protagonist_end: PlutchikVector,
    delivered_reader_end: PlutchikVector,
) -> SceneTargetReport:
    """Compare planned-end vs. delivered-end vectors.

    Used by the Emotion Critic. Thresholds:
        cosine_distance < 0.3  → on target
        0.3 ≤ d < 0.6          → drift (warn)
        d ≥ 0.6                → miss (error)
    """
    p_dist = plutchik.cosine_distance(target.protagonist_end, delivered_protagonist_end)
    r_dist = plutchik.cosine_distance(target.reader_end, delivered_reader_end)
    va_dist = valence_arousal_distance(
        plutchik_to_valence_arousal(target.reader_end),
        plutchik_to_valence_arousal(delivered_reader_end),
    )
    worst = max(p_dist, r_dist)

    if worst < 0.3:
        sev: str = "info"
        note = "On target."
        on_target = True
    elif worst < 0.6:
        sev = "warn"
        note = "Scene drifted from emotion target. Editor may want to tighten."
        on_target = False
    else:
        sev = "error"
        note = "Scene missed emotion target. Trigger rewrite."
        on_target = False

    return SceneTargetReport(
        protagonist_distance=round(p_dist, 3),
        reader_distance=round(r_dist, 3),
        va_distance=round(va_dist, 3),
        on_target=on_target,
        severity=sev,  # type: ignore[arg-type]
        note=note,
    )
