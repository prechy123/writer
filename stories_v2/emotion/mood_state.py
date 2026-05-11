"""Per-character mood state — evolving across scenes.

Each character carries:
    plutchik (8-axis emotion vector)
    valence / arousal (2D mood)
    last_event_summary
    last_updated_chapter / scene

A scene event is summarised by the Continuity v2 agent into a structured
delta (which axes intensified, which faded, plus a free-text summary).
This module applies that delta + a decay-toward-baseline transformation
to produce the next mood snapshot.
"""

from __future__ import annotations

from typing import Mapping, Optional

from ..schemas_v2 import (
    CharacterMoodSnapshot,
    PlutchikVector,
    SceneEmotionAxes,
)
from . import plutchik
from .valence_arousal import plutchik_to_valence_arousal


# Per-axis decay rate per scene. Anger fades fast; trust + anticipation are
# stickier. These are heuristic baselines tweakable per-character later
# (e.g., a vengeful character should decay anger more slowly).
_DEFAULT_DECAY = {
    "joy": 0.20,
    "trust": 0.08,
    "fear": 0.15,
    "surprise": 0.35,
    "sadness": 0.15,
    "disgust": 0.20,
    "anger": 0.25,
    "anticipation": 0.10,
}


def initial_mood(*, chapter_idx: int = 0, scene_idx: int = 0) -> CharacterMoodSnapshot:
    """A flat, low-intensity baseline. Story start."""
    return CharacterMoodSnapshot(
        chapter_idx=chapter_idx,
        scene_idx=scene_idx,
        plutchik=PlutchikVector(trust=0.3, anticipation=0.3),
        axes=SceneEmotionAxes(valence=0.1, arousal=0.4),
        last_event_summary="story_start",
    )


def decay_toward_baseline(
    current: PlutchikVector,
    *,
    baseline: Optional[PlutchikVector] = None,
    rate: Optional[Mapping[str, float]] = None,
) -> PlutchikVector:
    """Move current vector a fraction of the way toward the baseline.

    A character at boil at the end of scene N should partly cool by scene
    N+1 unless something keeps the heat on. Rate per axis is configurable.
    """
    baseline = baseline or PlutchikVector(trust=0.3, anticipation=0.3)
    rates = dict(_DEFAULT_DECAY)
    if rate:
        rates.update({k: v for k, v in rate.items() if v is not None})
    out = {}
    for ax in plutchik.AXES:
        cur = getattr(current, ax)
        base = getattr(baseline, ax)
        delta = (base - cur) * rates.get(ax, 0.15)
        out[ax] = max(0.0, min(1.0, cur + delta))
    return PlutchikVector(**out)


def apply_scene_event(
    snapshot: CharacterMoodSnapshot,
    *,
    delta: Mapping[str, float],
    chapter_idx: int,
    scene_idx: int,
    event_summary: str = "",
    baseline: Optional[PlutchikVector] = None,
    apply_decay: bool = True,
) -> CharacterMoodSnapshot:
    """Update mood with a scene-event delta, then optionally decay.

    Delta keys are Plutchik axis names mapped to signed floats in
    [-1, 1] — they are added BEFORE the decay step so a strong event
    fully lands before the next scene starts pulling the mood back to
    baseline.
    """
    new_plutchik = plutchik.from_mapping(
        {ax: getattr(snapshot.plutchik, ax) + float(delta.get(ax, 0.0)) for ax in plutchik.AXES}
    )
    if apply_decay:
        new_plutchik = decay_toward_baseline(new_plutchik, baseline=baseline)
    return CharacterMoodSnapshot(
        chapter_idx=chapter_idx,
        scene_idx=scene_idx,
        plutchik=new_plutchik,
        axes=plutchik_to_valence_arousal(new_plutchik),
        last_event_summary=event_summary or snapshot.last_event_summary,
    )


def summarize_mood(snapshot: CharacterMoodSnapshot, *, top_k: int = 2) -> str:
    """One-line human-readable summary of the current mood snapshot."""
    items = sorted(
        ((ax, getattr(snapshot.plutchik, ax)) for ax in plutchik.AXES),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top = [f"{ax}:{val:.2f}" for ax, val in items[:top_k] if val > 0.1]
    if not top:
        return "flat / neutral"
    return ", ".join(top)
