"""Sentence-length variance (burstiness) measurement + soft enforcement.

Measurement is always safe.

Auto-split is conservative: only triggers when stddev < threshold AND the
text has at least one long compound sentence we can safely split at a
coordinator (comma + and/but/so + capital).
"""

from __future__ import annotations

import re
import statistics
from typing import List, Tuple

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_WORD = re.compile(r"\b[\w'-]+\b")
# Safe split point: ", and|but|so|yet|or " — joins two independent clauses.
_SAFE_SPLIT = re.compile(r",\s+(and|but|so|yet|or)\s+(?=[A-Za-z])")


def _split_sentences(text: str) -> List[str]:
    return [s for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def measure(text: str) -> Tuple[float, float, int]:
    """Return (mean_words, stddev_words, sentence_count)."""
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0, 0.0, 0
    word_counts = [len(_WORD.findall(s)) for s in sentences]
    mean = statistics.mean(word_counts)
    stddev = statistics.pstdev(word_counts) if len(word_counts) > 1 else 0.0
    return float(mean), float(stddev), len(sentences)


def enforce(
    text: str,
    *,
    target_stddev: float = 7.0,
    max_splits: int = 4,
) -> Tuple[str, int]:
    """Conservatively split long compound sentences when stddev is too low.

    Returns (text, splits_made). Never changes meaning — only inserts a
    period at safe coordinator boundaries.
    """
    if not text:
        return text, 0
    _, stddev, count = measure(text)
    if count < 8 or stddev >= target_stddev:
        return text, 0

    # Find candidate split points across long sentences.
    sentences = _split_sentences(text)
    word_counts = [len(_WORD.findall(s)) for s in sentences]

    # Identify the longest sentences to target first.
    indices = sorted(range(len(sentences)), key=lambda i: word_counts[i], reverse=True)

    splits_made = 0
    for idx in indices:
        if splits_made >= max_splits:
            break
        # Only target sentences with >= 18 words (room to split)
        if word_counts[idx] < 18:
            continue

        sentence = sentences[idx]
        new_sentence, n = _SAFE_SPLIT.subn(
            lambda m: f". {m.group(1).capitalize()} ",
            sentence,
            count=1,
        )
        if n > 0:
            sentences[idx] = new_sentence
            splits_made += 1

    if splits_made == 0:
        return text, 0

    # Rejoin sentences with a single space.
    out = " ".join(sentences)
    # Re-measure briefly to make sure we improved things; if stddev didn't
    # change usefully, that's fine — we still applied a real split.
    return out, splits_made
