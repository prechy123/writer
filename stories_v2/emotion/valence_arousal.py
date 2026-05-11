"""Valence/Arousal 2D mood circumplex.

Russell's circumplex projects emotion into:
    valence  [-1, 1]   negative ↔ positive
    arousal  [0, 1]    calm ↔ activated

We project Plutchik → V/A via a fixed coefficient matrix derived from
the standard emotion-wheel valence/arousal mapping. This is a lossy
projection — the Plutchik vector remains the source of truth — but V/A
gives the Pacing critic + Reader Arc planner a low-dim handle.
"""

from __future__ import annotations

import math
from typing import Tuple

from ..schemas_v2 import PlutchikVector, SceneEmotionAxes

# Coefficients drawn from the standard emotion → V/A mapping
# (Russell 1980 / Bradley & Lang norms, condensed).
# axis: (valence_weight, arousal_weight)
_VA_COEFFS = {
    "joy":          (+1.0, +0.55),
    "trust":        (+0.7, +0.20),
    "fear":         (-0.6, +0.95),
    "surprise":     (+0.1, +0.85),
    "sadness":      (-0.85, +0.20),
    "disgust":      (-0.75, +0.55),
    "anger":        (-0.65, +0.95),
    "anticipation": (+0.4, +0.65),
}


def plutchik_to_valence_arousal(v: PlutchikVector) -> SceneEmotionAxes:
    """Project Plutchik → valence in [-1,1], arousal in [0,1]."""
    val = 0.0
    aro = 0.0
    weight_sum = 0.0
    for ax, (vw, aw) in _VA_COEFFS.items():
        x = getattr(v, ax)
        val += x * vw
        aro += x * aw
        weight_sum += x
    if weight_sum > 0:
        val /= max(weight_sum, 1.0)
        aro /= max(weight_sum, 1.0)
    return SceneEmotionAxes(
        valence=max(-1.0, min(1.0, val)),
        arousal=max(0.0, min(1.0, aro)),
    )


def valence_arousal_distance(a: SceneEmotionAxes, b: SceneEmotionAxes) -> float:
    """Euclidean distance in the V/A plane.  Range ~[0, sqrt(5)]."""
    dv = a.valence - b.valence
    da = a.arousal - b.arousal
    return math.sqrt(dv * dv + da * da)
