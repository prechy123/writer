"""Em-dash strip + AI-phrase blocklist.

This module never asks the LLM. It does the most important work in the
v2 stack: removes the single signal that AI detectors weight most
heavily.

We replace em-dashes contextually:
  - Mid-sentence interruption  → period + capitalise the next word
  - Parenthetical aside        → commas
  - Compound break             → semicolon NOT used (also a tell);
                                  prefer period + capital
"""

from __future__ import annotations

import re
from typing import List, Tuple


# All variants of the em-dash glyph, plus the rarer minus / quotation dash.
_DASH_RE = re.compile(r"[—―−⸺⸻]")

# Word-character followed by dash followed by word-character — likely a
# mid-sentence interruption that wants a period+capital.
_INTERRUPT_RE = re.compile(r"(\w)[—―−⸺⸻](\w)")

# Sentence-end dash glyph followed by " word" — usually wants a period+capital.
_SENTENCE_END_RE = re.compile(r"([\.!\?])[\s]*[—―−⸺⸻][\s]+(\w)")

# Phrase blocklist. Conservative substitutions chosen so deletion never leaves
# a sentence dangling. Where no clean substitute exists, the word becomes the
# empty string (the surrounding context still reads).
PHRASE_BLOCKLIST: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(delves?|delving|delved)\s+into\b", re.IGNORECASE), "looks into"),
    (re.compile(r"\b(tapestry of\s+\w+)\b", re.IGNORECASE), ""),
    (re.compile(r"\bnavigate(?:s|d|ing)?\s+the\s+complexit(?:y|ies)\s+of\b", re.IGNORECASE), "work through"),
    (re.compile(r"\bunwavering\b", re.IGNORECASE), "steady"),
    (re.compile(r"\bis\s+a\s+testament\s+to\b", re.IGNORECASE), "shows"),
    (re.compile(r"\b(it\'s\s+important\s+to\s+note\s+that|in\s+conclusion,?|at\s+its\s+core,?)\b", re.IGNORECASE), ""),
    (re.compile(r"\bmyriad\b", re.IGNORECASE), "many"),
    (re.compile(r"\bbustling\b", re.IGNORECASE), "busy"),
    (re.compile(r"\bnestled\b", re.IGNORECASE), "tucked"),
    (re.compile(r"\bgleaming\b", re.IGNORECASE), "bright"),
    (re.compile(r"\bpristine\b", re.IGNORECASE), "clean"),
]


def strip_em_dashes(text: str) -> Tuple[str, int]:
    """Return (cleaned_text, replacements_made)."""
    if not text or "—" not in text and not _DASH_RE.search(text):
        return text, 0

    count = len(_DASH_RE.findall(text))

    # Pattern 1: sentence-terminator + dash + capital -> drop the dash, keep period
    text = _SENTENCE_END_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}", text)

    # Pattern 2: word—word -> word. Word
    def _interrupt(m: re.Match) -> str:
        return f"{m.group(1)}. {m.group(2).upper()}"

    text = _INTERRUPT_RE.sub(_interrupt, text)

    # Any leftover dashes — replace with a simple comma fallback. (Some occur
    # in lists or attribution; comma is the safest universal substitute.)
    text = _DASH_RE.sub(", ", text)

    # Collapse any double-spaces we introduced.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text, count


def strip_blocklist(text: str) -> Tuple[str, int]:
    """Replace banned phrases in-place. Returns (cleaned, strikes_count)."""
    if not text:
        return text, 0
    strikes = 0
    for pattern, replacement in PHRASE_BLOCKLIST:
        new_text, n = pattern.subn(replacement, text)
        if n > 0:
            strikes += n
            text = new_text
    # Clean up any double-spaces or "  ." artifacts left by empty replacements.
    text = re.sub(r"\s+([,.;:!\?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\.\s*\.+", ".", text)
    return text, strikes
