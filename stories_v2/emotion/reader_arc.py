"""Reader emotion arc planning.

What the reader is *meant to feel* at each beat. This is decoupled from
character mood — a scene where a character is at their lowest can be
exhilarating for the reader, and vice versa.

The 4-act web-novel arc gets default reader emotional targets per act;
chapters within an act interpolate; scenes within a chapter follow
Kishōtenketsu (Introduction → Development → Twist → Conclusion).
"""

from __future__ import annotations

from typing import Dict, List

from ..schemas_v2 import PlutchikVector

# Each act gets a target reader vector (rough).
# These are starting defaults the Architect prompt can override per story.
_ARC_DEFAULT_TARGETS: Dict[str, PlutchikVector] = {
    "discovery":   PlutchikVector(anticipation=0.65, trust=0.35, surprise=0.30, joy=0.25),
    "escalation":  PlutchikVector(anticipation=0.70, fear=0.45, anger=0.30, surprise=0.35),
    "revelation":  PlutchikVector(surprise=0.75, fear=0.55, sadness=0.35, anger=0.40),
    "catharsis":   PlutchikVector(joy=0.55, trust=0.55, anticipation=0.40, sadness=0.20),
}

# Kishōtenketsu phases per chapter (4 scenes). For 3-scene chapters we fold
# Development into Introduction; for 5/6-scene chapters we duplicate
# Development.
_KISHO_PHASES = ["introduction", "development", "twist", "conclusion"]


def arc_act_targets() -> Dict[str, PlutchikVector]:
    """Return the default per-act reader emotion targets (immutable copy)."""
    return {k: PlutchikVector(**v.model_dump()) for k, v in _ARC_DEFAULT_TARGETS.items()}


def chapter_reader_arc(
    *,
    act: str,
    chapter_position: float,
    cliffhanger_intensity: str = "high",
) -> Dict[str, PlutchikVector]:
    """Reader emotion trajectory across one chapter's scenes.

    ``chapter_position`` in [0, 1] is where this chapter sits inside its
    act — 0 means start of act, 1 means end. Used to interpolate intensity.

    ``cliffhanger_intensity`` ratchets the end-of-chapter vector.

    Returns dict of {phase_name: PlutchikVector}.
    """
    act_target = _ARC_DEFAULT_TARGETS.get(act, _ARC_DEFAULT_TARGETS["discovery"])
    cliff_boost = {"low": 0.95, "medium": 1.15, "high": 1.35}.get(cliffhanger_intensity, 1.15)

    intensity_intro = 0.4 + 0.3 * chapter_position
    intensity_dev = 0.6 + 0.3 * chapter_position
    intensity_twist = 0.8 + 0.2 * chapter_position
    intensity_concl = 0.9 + 0.1 * chapter_position

    out: Dict[str, PlutchikVector] = {}
    for phase, scale in zip(
        _KISHO_PHASES,
        [intensity_intro, intensity_dev, intensity_twist, intensity_concl],
    ):
        eff = scale
        if phase == "conclusion":
            eff = min(1.0, scale * cliff_boost)
        out[phase] = PlutchikVector(
            **{
                ax: max(0.0, min(1.0, getattr(act_target, ax) * eff))
                for ax in PlutchikVector.model_fields
            }
        )
    return out


def scene_reader_target(
    chapter_arc: Dict[str, PlutchikVector],
    *,
    scene_idx: int,
    total_scenes: int,
) -> PlutchikVector:
    """Pick a target vector for one scene given the chapter's reader arc.

    Maps a 3-6 scene chapter onto the 4 Kishōtenketsu phases.
    """
    total = max(1, total_scenes)
    # Distribute scenes across phases. With 4 scenes -> 1:1. With 3 -> intro+dev merged.
    if total <= 3:
        phase_index = min(3, scene_idx)
        if scene_idx == 0:
            phase_name = "introduction"
        elif scene_idx == total - 1:
            phase_name = "conclusion"
        else:
            phase_name = "twist"
    else:
        ratio = (scene_idx + 0.5) / total
        if ratio < 0.25:
            phase_name = "introduction"
        elif ratio < 0.55:
            phase_name = "development"
        elif ratio < 0.85:
            phase_name = "twist"
        else:
            phase_name = "conclusion"
    return chapter_arc.get(phase_name, list(chapter_arc.values())[-1])


def all_act_keys() -> List[str]:
    return list(_ARC_DEFAULT_TARGETS.keys())
