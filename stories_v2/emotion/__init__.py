"""Emotion engine — character mood vectors + reader emotion arc.

Two coordinate systems:
    Plutchik 8-axis (joy, trust, fear, surprise, sadness, disgust, anger,
        anticipation) — used for character mood state.
    Valence/Arousal 2D circumplex — used for reader emotion arc and as
        a low-dimensional axis the Pacing critic can reason about.

Modules:
    plutchik         — vector ops (add, scale, blend, distance, normalise)
    valence_arousal  — 2D mood circumplex; project Plutchik → V/A
    mood_state       — per-character evolving mood, scene-driven transitions
    reader_arc       — per-scene / per-chapter / per-arc reader emotion targets
    targets          — SceneEmotionTarget builder + classification helpers
"""

from .mood_state import (
    apply_scene_event,
    decay_toward_baseline,
    initial_mood,
    summarize_mood,
)
from .plutchik import (
    add as plutchik_add,
    blend as plutchik_blend,
    cosine_distance as plutchik_cosine_distance,
    delta as plutchik_delta,
    intensity,
    normalize as plutchik_normalize,
    scale as plutchik_scale,
)
from .reader_arc import (
    arc_act_targets,
    chapter_reader_arc,
    scene_reader_target,
)
from .targets import (
    SceneEmotionTarget,
    build_scene_target,
    score_target_delivery,
)
from .valence_arousal import plutchik_to_valence_arousal, valence_arousal_distance

__all__ = [
    # plutchik
    "plutchik_add",
    "plutchik_blend",
    "plutchik_cosine_distance",
    "plutchik_delta",
    "plutchik_normalize",
    "plutchik_scale",
    "intensity",
    # valence/arousal
    "plutchik_to_valence_arousal",
    "valence_arousal_distance",
    # mood
    "apply_scene_event",
    "decay_toward_baseline",
    "initial_mood",
    "summarize_mood",
    # arc
    "arc_act_targets",
    "chapter_reader_arc",
    "scene_reader_target",
    # targets
    "SceneEmotionTarget",
    "build_scene_target",
    "score_target_delivery",
]
