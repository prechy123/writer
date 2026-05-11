"""Pick the top-K exemplars for a scene profile.

Scoring is tag-overlap based — fast, deterministic, no embeddings.
For more sophisticated retrieval, plug in a vector store later (the
SceneWriter doesn't need it; tag-based matches are tight enough).
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from .loader import list_exemplars
from .schema import CorpusEntry


def _score(entry: CorpusEntry, *, genres, techniques, emotion_tags, pov) -> int:
    score = 0
    g_set = set(x.lower() for x in genres)
    t_set = set(x.lower() for x in techniques)
    e_set = set(x.lower() for x in emotion_tags)

    entry_g = set(x.lower() for x in entry.genres)
    entry_t = set(x.lower() for x in entry.techniques)
    entry_e = set(x.lower() for x in entry.emotion_tags)

    score += len(entry_g & g_set) * 3
    score += len(entry_t & t_set) * 4
    score += len(entry_e & e_set) * 2
    if "generic" in entry_g:
        score += 1
    if pov and entry.pov == pov:
        score += 2
    return score


def pick_exemplars(
    *,
    genres: Iterable[str] = (),
    techniques: Iterable[str] = (),
    emotion_tags: Iterable[str] = (),
    pov: Optional[str] = None,
    k: int = 3,
) -> List[CorpusEntry]:
    """Return up to K exemplars ranked by tag-overlap to the scene profile."""
    candidates = list_exemplars()
    if not candidates:
        return []
    scored = [
        (_score(e, genres=genres, techniques=techniques, emotion_tags=emotion_tags, pov=pov), e)
        for e in candidates
    ]
    scored = [pair for pair in scored if pair[0] > 0]
    scored.sort(key=lambda p: p[0], reverse=True)
    return [e for _, e in scored[:k]]
