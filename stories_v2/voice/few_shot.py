"""Few-shot block builders.

The Scene Writer injects 2–3 sample lines per active character so the
LLM has concrete voice anchors. This module formats those anchors into
a compact, model-friendly block.

Rule of thumb: never inject more than 3 sample_lines per character per
scene — past that, the model starts pattern-matching too literally.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Optional

from ..schemas_v2 import CharacterBibleV2, VoiceFingerprint


def build_few_shot_block(
    *,
    name: str,
    fingerprint: VoiceFingerprint,
    max_lines: int = 3,
) -> str:
    """One character's anchor block.

    Format keeps the LLM's attention on idiolect, not on the meta
    framing — short, dense, no XML, no JSON.
    """
    if not fingerprint or not fingerprint.sample_lines:
        return ""
    lines = list(fingerprint.sample_lines)
    if len(lines) > max_lines:
        lines = random.sample(lines, max_lines)
    rendered = "\n".join(f'  - "{l.strip()}"' for l in lines if l.strip())
    pieces = [f"{name} speaks like this:", rendered]

    extras = []
    if fingerprint.preferred_phrases:
        extras.append(
            f"  signature phrases: {', '.join(fingerprint.preferred_phrases[:4])}"
        )
    if fingerprint.verbal_tics:
        extras.append(f"  verbal tics: {', '.join(fingerprint.verbal_tics[:3])}")
    if fingerprint.banned_phrases:
        extras.append(
            f"  NEVER says: {', '.join(fingerprint.banned_phrases[:4])}"
        )
    if extras:
        pieces.append("\n".join(extras))
    return "\n".join(pieces)


def build_scene_few_shot(
    present_characters: Iterable[CharacterBibleV2],
    *,
    max_characters: int = 5,
    max_lines_per_character: int = 3,
) -> str:
    """Build the full few-shot block for one scene.

    Side-tier characters are skipped if there are too many characters to
    fit; main/recurring are prioritised.
    """
    chars: List[CharacterBibleV2] = list(present_characters)
    chars.sort(key=lambda c: ({"main": 0, "recurring": 1, "side": 2}.get(c.tier.value, 9), c.name))

    blocks: List[str] = []
    for ch in chars[:max_characters]:
        if not ch.voice_fingerprint:
            continue
        block = build_few_shot_block(
            name=ch.name,
            fingerprint=ch.voice_fingerprint,
            max_lines=max_lines_per_character,
        )
        if block:
            blocks.append(block)
    if not blocks:
        return ""
    return "\n\n".join(blocks)
