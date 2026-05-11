"""Plutchik 8-axis emotion vector operations.

Axes (Plutchik's wheel of emotions, primary 8):
    joy, trust, fear, surprise, sadness, disgust, anger, anticipation

All operations work on ``PlutchikVector`` model instances and return
new instances — immutable semantics keep mood-history snapshots clean.

Intensity is on [0, 1]. Operations that would push a value out of range
are clamped, not wrapped or saturated softly (we want clearly-bounded
deltas that the Emotion Critic can compare).
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

from ..schemas_v2 import PlutchikVector

AXES = (
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def add(a: PlutchikVector, b: PlutchikVector) -> PlutchikVector:
    return PlutchikVector(**{ax: _clamp(getattr(a, ax) + getattr(b, ax)) for ax in AXES})


def scale(v: PlutchikVector, factor: float) -> PlutchikVector:
    return PlutchikVector(**{ax: _clamp(getattr(v, ax) * factor) for ax in AXES})


def blend(a: PlutchikVector, b: PlutchikVector, weight_b: float = 0.5) -> PlutchikVector:
    """Convex combination. weight_b in [0,1]."""
    weight_b = _clamp(weight_b)
    weight_a = 1.0 - weight_b
    return PlutchikVector(
        **{ax: _clamp(getattr(a, ax) * weight_a + getattr(b, ax) * weight_b) for ax in AXES}
    )


def delta(a: PlutchikVector, b: PlutchikVector) -> Mapping[str, float]:
    """b - a, axis by axis (NOT clamped — deltas are signed)."""
    return {ax: getattr(b, ax) - getattr(a, ax) for ax in AXES}


def cosine_distance(a: PlutchikVector, b: PlutchikVector) -> float:
    """Cosine distance (1 - cosine similarity) in [0, 2].

    Used by the Emotion Critic to compare planned-end vs. delivered-end
    Plutchik vectors. < 0.3 = on target; 0.3-0.6 = drift; > 0.6 = miss.
    """
    a_vals = [getattr(a, ax) for ax in AXES]
    b_vals = [getattr(b, ax) for ax in AXES]
    dot = sum(x * y for x, y in zip(a_vals, b_vals))
    na = math.sqrt(sum(x * x for x in a_vals))
    nb = math.sqrt(sum(y * y for y in b_vals))
    if na == 0.0 or nb == 0.0:
        # When at least one vector is all-zero, define distance as 1.0 (orthogonal)
        # so the critic flags it rather than mistaking it for a perfect match.
        return 1.0
    return 1.0 - dot / (na * nb)


def intensity(v: PlutchikVector) -> float:
    """L2 norm scaled to [0, 1] (assuming each axis in [0,1])."""
    s = sum(getattr(v, ax) ** 2 for ax in AXES)
    return min(1.0, math.sqrt(s / len(AXES)))


def normalize(v: PlutchikVector, target_intensity: float = 1.0) -> PlutchikVector:
    """Rescale so the L2 norm matches ``target_intensity``. Zero stays zero."""
    cur = intensity(v)
    if cur == 0.0:
        return v
    factor = target_intensity / cur
    return scale(v, factor)


def from_mapping(m: Mapping[str, float]) -> PlutchikVector:
    """Build a PlutchikVector from a partial dict (missing axes default to 0)."""
    return PlutchikVector(**{ax: _clamp(float(m.get(ax, 0.0))) for ax in AXES})


def aggregate(vectors: Iterable[PlutchikVector]) -> PlutchikVector:
    """Mean of a sequence (returns zero vector if empty)."""
    vs = list(vectors)
    if not vs:
        return PlutchikVector()
    n = float(len(vs))
    return PlutchikVector(
        **{ax: sum(getattr(v, ax) for v in vs) / n for ax in AXES}
    )
