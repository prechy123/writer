"""Inject contractions in dialogue.

LLMs are often weirdly formal in spoken dialogue ("I do not", "you are
not"). We boost the contraction rate inside quoted dialogue by safely
substituting the most common patterns. We do NOT touch narration —
contractions in narration are style-dependent; contractions in dialogue
are baseline expected.

Conservative: we only contract when the word boundaries are unambiguous
and the substitution doesn't change meaning (no "she'd" for "she had"
when "she would" is intended — we keep those out).
"""

from __future__ import annotations

import re
from typing import List, Tuple

# Dialogue spans inside double quotes.
_DIALOGUE_RE = re.compile(r"\"([^\"\n]+)\"")

# Safe contractions: subject + auxiliary that always contract the same way
# regardless of meaning.
_CONTRACTIONS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bdo not\b", re.IGNORECASE), "don't"),
    (re.compile(r"\bdoes not\b", re.IGNORECASE), "doesn't"),
    (re.compile(r"\bdid not\b", re.IGNORECASE), "didn't"),
    (re.compile(r"\bis not\b", re.IGNORECASE), "isn't"),
    (re.compile(r"\bare not\b", re.IGNORECASE), "aren't"),
    (re.compile(r"\bwas not\b", re.IGNORECASE), "wasn't"),
    (re.compile(r"\bwere not\b", re.IGNORECASE), "weren't"),
    (re.compile(r"\bhas not\b", re.IGNORECASE), "hasn't"),
    (re.compile(r"\bhave not\b", re.IGNORECASE), "haven't"),
    (re.compile(r"\bhad not\b", re.IGNORECASE), "hadn't"),
    (re.compile(r"\bcannot\b", re.IGNORECASE), "can't"),
    (re.compile(r"\bcan not\b", re.IGNORECASE), "can't"),
    (re.compile(r"\bwill not\b", re.IGNORECASE), "won't"),
    (re.compile(r"\bshould not\b", re.IGNORECASE), "shouldn't"),
    (re.compile(r"\bwould not\b", re.IGNORECASE), "wouldn't"),
    (re.compile(r"\bcould not\b", re.IGNORECASE), "couldn't"),
    (re.compile(r"\bmust not\b", re.IGNORECASE), "mustn't"),
    (re.compile(r"\bI am\b"), "I'm"),
    (re.compile(r"\byou are\b", re.IGNORECASE), "you're"),
    (re.compile(r"\bwe are\b", re.IGNORECASE), "we're"),
    (re.compile(r"\bthey are\b", re.IGNORECASE), "they're"),
    (re.compile(r"\bit is\b", re.IGNORECASE), "it's"),
    (re.compile(r"\bthat is\b", re.IGNORECASE), "that's"),
    (re.compile(r"\bthere is\b", re.IGNORECASE), "there's"),
    (re.compile(r"\bI have\b", re.IGNORECASE), "I've"),
    (re.compile(r"\byou have\b", re.IGNORECASE), "you've"),
    (re.compile(r"\bwe have\b", re.IGNORECASE), "we've"),
    (re.compile(r"\bthey have\b", re.IGNORECASE), "they've"),
    (re.compile(r"\bI will\b"), "I'll"),
    (re.compile(r"\byou will\b", re.IGNORECASE), "you'll"),
    (re.compile(r"\bhe will\b", re.IGNORECASE), "he'll"),
    (re.compile(r"\bshe will\b", re.IGNORECASE), "she'll"),
    (re.compile(r"\bthey will\b", re.IGNORECASE), "they'll"),
    (re.compile(r"\bwe will\b", re.IGNORECASE), "we'll"),
    (re.compile(r"\blet us\b", re.IGNORECASE), "let's"),
]


def inject_in_dialogue(text: str) -> Tuple[str, int]:
    """Replace formal forms inside dialogue. Returns (text, injections)."""
    if not text or '"' not in text:
        return text, 0

    total_injected = 0

    def _replace(match: re.Match) -> str:
        nonlocal total_injected
        inner = match.group(1)
        modified = inner
        for pattern, sub in _CONTRACTIONS:
            modified, n = pattern.subn(sub, modified)
            total_injected += n
        return f"\"{modified}\""

    text = _DIALOGUE_RE.sub(_replace, text)
    return text, total_injected
