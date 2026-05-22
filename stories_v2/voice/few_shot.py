"""Few-shot block builders.

The Scene Writer injects 2–3 sample lines per active character so the
LLM has concrete voice anchors. This module formats those anchors into
a compact, model-friendly block.

Rotation policy (important):
  - If a character has 3+ sample_lines, pick a deterministic but
    chapter+scene-rotated subset so the writer doesn't see the same
    line every scene.
  - If a character has only 1-2 sample_lines, show them in the FIRST
    scene of a chapter, then SKIP the few-shot block for that character
    on subsequent scenes — re-injecting the same line on every scene
    is the single biggest cause of verbatim phrase repetition across a
    chapter (the writer pattern-matches and pastes).
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional

from ..schemas_v2 import CharacterBibleV2, VoiceFingerprint


def _stable_pick(
    candidates: List[str],
    *,
    k: int,
    seed: int,
) -> List[str]:
    """Deterministic rotating selection.

    Picks ``k`` candidates by walking the list starting at ``seed % len(candidates)``.
    Same seed → same selection; nearby seeds → adjacent picks. Gives us
    rotation without randomness so tests stay reproducible.
    """
    if not candidates:
        return []
    n = len(candidates)
    if k >= n:
        return list(candidates)
    start = seed % n
    return [candidates[(start + i) % n] for i in range(k)]


def _scene_seed(name: str, chapter_idx: Optional[int], scene_idx: Optional[int]) -> int:
    """A stable integer seed from (character name, chapter, scene)."""
    parts = f"{name}|{chapter_idx or 0}|{scene_idx or 0}".encode("utf-8")
    return int(hashlib.sha1(parts).hexdigest()[:8], 16)


def build_few_shot_block(
    *,
    name: str,
    fingerprint: VoiceFingerprint,
    max_lines: int = 3,
    chapter_idx: Optional[int] = None,
    scene_idx: Optional[int] = None,
) -> str:
    """One character's anchor block.

    Format keeps the LLM's attention on idiolect, not on the meta
    framing — short, dense, no XML, no JSON.
    """
    if not fingerprint or not fingerprint.sample_lines:
        return ""
    lines = [l.strip() for l in fingerprint.sample_lines if l and l.strip()]
    if not lines:
        return ""

    # If the character only has 1-2 lines, show them ONLY on the first
    # scene of the chapter to avoid verbatim repetition downstream.
    if len(lines) < 3 and scene_idx is not None and scene_idx > 0:
        # Skip the sample_lines section; carry only fingerprint metadata
        # so the writer still has SOME voice signal.
        rendered = ""
    else:
        seed = _scene_seed(name, chapter_idx, scene_idx)
        picked = _stable_pick(lines, k=min(max_lines, len(lines)), seed=seed)
        rendered = "\n".join(f'  - "{l}"' for l in picked)

    pieces: List[str]
    if rendered:
        pieces = [f"{name} speaks like this:", rendered]
    else:
        pieces = [f"{name} (voice carries from prior scenes; rotate phrasing, do NOT repeat earlier lines):"]

    extras: List[str] = []
    if fingerprint.preferred_phrases:
        # Rotate preferred phrases too.
        prefs = list(fingerprint.preferred_phrases)
        if prefs:
            seed = _scene_seed(name + ":pref", chapter_idx, scene_idx)
            prefs = _stable_pick(prefs, k=min(3, len(prefs)), seed=seed)
            extras.append(f"  signature phrases (use sparingly): {', '.join(prefs)}")
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
    chapter_idx: Optional[int] = None,
    scene_idx: Optional[int] = None,
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
            chapter_idx=chapter_idx,
            scene_idx=scene_idx,
        )
        if block:
            blocks.append(block)
    if not blocks:
        return ""
    return "\n\n".join(blocks)
