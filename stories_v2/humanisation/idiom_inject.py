"""Soft pass: surface character-specific idioms in their dialogue.

We don't rewrite dialogue here — that's the Editor's job. We just count
how often each main character's preferred phrases / catchphrases / verbal
tics appear in their dialogue, and emit a note for any character that
has substantial dialogue (> N words) yet zero idiom hits. The Voice
Critic + Editor pick up the slack on the next cycle.
"""

from __future__ import annotations

import re
from typing import Dict, List

from ..schemas_v2 import CharacterBibleV2

_DIALOGUE_RE = re.compile(r"\"([^\"\n]+)\"")


def audit(text: str, present_characters: List[CharacterBibleV2]) -> List[str]:
    """Return list of notes (strings) describing missing idiom signals."""
    if not text or not present_characters:
        return []
    dialogue_words = []
    for m in _DIALOGUE_RE.finditer(text):
        dialogue_words.extend(re.findall(r"\b[\w'-]+\b", m.group(1)))
    if not dialogue_words:
        return []
    joined_lower = " ".join(w.lower() for w in dialogue_words)

    notes: List[str] = []
    for ch in present_characters:
        if not ch.voice_fingerprint:
            continue
        vf = ch.voice_fingerprint
        anchors = [p.lower() for p in (vf.preferred_phrases or []) + (vf.catchphrases or []) + (vf.verbal_tics or [])]
        anchors = [a for a in anchors if a]
        if not anchors:
            continue
        hits = sum(1 for a in anchors if a in joined_lower)
        if hits == 0 and len(dialogue_words) > 50:
            notes.append(
                f"character '{ch.name}' has anchor phrases ({anchors[:3]}) but none appeared in dialogue."
            )
    return notes
