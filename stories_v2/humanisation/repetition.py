"""Cross-scene repetition detector.

When the same n-gram phrase shows up verbatim across multiple scenes in
the same story, that's the loudest possible AI-tell. Real writers find
new ways to say the same thing; LLMs anchor on whatever phrasing landed
last time and reuse it.

This module compares a fresh scene draft against prior committed scenes
in the same story and either:
  - reports the repeats so the editor / deepener pass can rewrite them
  - strips them outright when they're standalone enough to delete

Heuristics:
  - n-gram size 5 (longer than common idioms, shorter than full sentences)
  - case-insensitive after collapsing whitespace
  - skips dialogue (inside double quotes) — characters legitimately repeat
  - skips very common sequences (would over-flag "out of the corner of his eye")
  - looks at last 4 scenes max (same chapter + last scene of prior chapter)
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_WORD = re.compile(r"[A-Za-z']+")
_DIALOGUE_SPAN = re.compile(r'"[^"\n]*"')
# Sequences common enough we don't want to flag them even if repeated.
_STOPWORD_HEAVY = re.compile(r"^(?:the|a|an|of|in|on|at|to|for|from|with|and|or|but|that|this|his|her|their|its|he|she|they|it)\s")


def _normalise(text: str) -> str:
    """Lowercase, strip dialogue, collapse whitespace."""
    no_dialogue = _DIALOGUE_SPAN.sub(" ", text or "")
    tokens = _WORD.findall(no_dialogue.lower())
    return " ".join(tokens)


def _ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _is_substantive(phrase: str) -> bool:
    """Filter out boilerplate n-grams."""
    if _STOPWORD_HEAVY.match(phrase):
        # Allow if the middle words are content-heavy (>= 2 nouns/verbs).
        # Cheaper proxy: at least 3 words longer than 4 characters.
        long_words = [w for w in phrase.split() if len(w) > 4]
        if len(long_words) < 2:
            return False
    # Drop n-grams that are mostly numbers or punctuation residue.
    if sum(1 for c in phrase if c.isalpha()) < len(phrase) * 0.6:
        return False
    return True


def build_prior_phrase_index(prior_scenes_prose: List[str], *, n: int = 5) -> Dict[str, int]:
    """Build a frequency index of n-grams from prior scenes.

    Returns ``{phrase: count}`` where count is how many prior scenes the
    phrase appeared in (not occurrences). Repeating across MULTIPLE prior
    scenes is the strong signal.
    """
    index: Dict[str, int] = {}
    for prose in prior_scenes_prose:
        normalised = _normalise(prose)
        seen_in_this_scene = set()
        for gram in _ngrams(normalised.split(), n):
            if not _is_substantive(gram):
                continue
            if gram in seen_in_this_scene:
                continue
            seen_in_this_scene.add(gram)
            index[gram] = index.get(gram, 0) + 1
    return index


def find_repeats(
    current_prose: str,
    *,
    prior_index: Dict[str, int],
    n: int = 5,
    min_prior_scenes: int = 1,
) -> List[Tuple[str, int]]:
    """Return ``[(phrase, prior_scene_count), ...]`` for phrases that the
    current scene repeats from the prior index.
    """
    if not current_prose or not prior_index:
        return []
    normalised = _normalise(current_prose)
    seen = set()
    repeats: List[Tuple[str, int]] = []
    for gram in _ngrams(normalised.split(), n):
        if gram in seen:
            continue
        seen.add(gram)
        prior_count = prior_index.get(gram, 0)
        if prior_count >= min_prior_scenes and _is_substantive(gram):
            repeats.append((gram, prior_count))
    # Sort by prior frequency desc — the most-repeated land first.
    repeats.sort(key=lambda p: p[1], reverse=True)
    return repeats


def strip_obvious_repeats(
    prose: str,
    *,
    repeats: List[Tuple[str, int]],
    max_strips: int = 4,
) -> Tuple[str, int]:
    """Conservatively remove sentences that are anchored on a repeated phrase.

    We don't blindly delete the phrase — we delete the whole sentence
    only when the sentence's main spine matches an n-gram that repeated
    >= 2 prior scenes. This catches the verbatim "Give me a sec, I'll
    reroute the sensor feed" / "The Core's heart beats beneath us"
    pattern without nuking legitimate motifs.

    Returns ``(cleaned_prose, sentences_dropped)``.
    """
    if not prose or not repeats:
        return prose, 0

    # Only act on phrases that landed in >= 2 prior scenes.
    aggressive = [phrase for phrase, count in repeats if count >= 2]
    if not aggressive:
        return prose, 0

    sentences = re.split(r"(?<=[.!?])\s+", prose)
    kept: List[str] = []
    dropped = 0

    for sentence in sentences:
        if dropped >= max_strips:
            kept.append(sentence)
            continue
        norm = " ".join(_WORD.findall(sentence.lower()))
        # Skip dialogue sentences — character speech can legitimately repeat.
        if '"' in sentence and sum(1 for c in sentence if c == '"') >= 2:
            kept.append(sentence)
            continue
        match_hit = False
        for phrase in aggressive:
            if phrase in norm:
                match_hit = True
                break
        if match_hit:
            dropped += 1
            continue
        kept.append(sentence)

    if dropped == 0:
        return prose, 0
    cleaned = " ".join(kept).strip()
    return cleaned, dropped


def load_prior_scenes_prose(story_id: str, *, chapter_idx: int, scene_idx: int, lookback: int = 6) -> List[str]:
    """Fetch up to ``lookback`` prior committed scenes from Mongo.

    Looks at: all scenes in the current chapter before ``scene_idx`` +
    the trailing scenes of the previous chapter. Imported chapters count.
    """
    # Local import to avoid circular imports at module load time.
    from .. import mongo

    prior: List[str] = []

    # Same chapter, earlier scenes.
    if scene_idx > 0:
        cursor = mongo.col(mongo.COL_SCENES).find(
            {"story_id": story_id, "chapter_idx": chapter_idx, "scene_idx": {"$lt": scene_idx}},
            projection={"scene_idx": 1, "final_prose": 1},
        ).sort("scene_idx", 1)
        for doc in cursor:
            prose = doc.get("final_prose") or ""
            if prose:
                prior.append(prose)

    # Previous chapters (newest first), until lookback is exhausted.
    remaining = max(0, lookback - len(prior))
    if remaining > 0 and chapter_idx > 0:
        cursor = mongo.col(mongo.COL_SCENES).find(
            {"story_id": story_id, "chapter_idx": {"$lt": chapter_idx}},
            projection={"chapter_idx": 1, "scene_idx": 1, "final_prose": 1},
        ).sort([("chapter_idx", -1), ("scene_idx", -1)]).limit(remaining)
        for doc in cursor:
            prose = doc.get("final_prose") or ""
            if prose:
                prior.append(prose)

    return prior
