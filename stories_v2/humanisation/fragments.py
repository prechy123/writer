"""Detect sentence-fragment density.

Auto-inserting fragments is risky (the writer's flow can be off), so
this module only measures. If fragments are absent, the Pacing critic
flags it on the next round; the Editor learns to add them.
"""

from __future__ import annotations

import re
from typing import Tuple

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_WORD = re.compile(r"\b[\w'-]+\b")


def count_fragments(text: str) -> Tuple[int, int]:
    """Return (fragment_count, total_sentences).

    A "fragment" here is heuristic: any sentence with <= 3 words OR with
    no main verb pattern. We approximate "no main verb" by lacking any
    of the common auxiliary / be verbs in their conjugated forms.
    """
    if not text:
        return 0, 0
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return 0, 0
    fragment_count = 0
    for s in sentences:
        words = _WORD.findall(s)
        if len(words) <= 3:
            fragment_count += 1
            continue
        # Quick "has verb-like word" check — extremely permissive
        lowered = s.lower()
        has_be = bool(re.search(r"\b(is|are|was|were|am|be|been|being)\b", lowered))
        has_aux = bool(re.search(r"\b(have|has|had|do|does|did|will|would|could|should|may|might|must|can|shall)\b", lowered))
        # Crude "verb-y" heuristic: ends in 'ed', 'ing', or 's' on a non-determiner
        has_inflected = bool(re.search(r"\b\w+(?:ed|ing|s)\b", lowered))
        if not (has_be or has_aux or has_inflected):
            fragment_count += 1
    return fragment_count, len(sentences)
