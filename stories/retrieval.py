"""Embedding + retrieval helpers for chapter-summary RAG.

The Writer's prompt cannot fit every prior chapter's summary once a book
crosses ~30 chapters — the ``running_summary`` string used to grow
linearly. This module replaces that with a top-K nearest-neighbour
retrieval over per-chapter summary embeddings stored on each
``chapter_metadata`` entry.

Design notes:

- Embeddings are produced via Together AI (same provider as the LLMs).
- Storage is inline on each chapter_metadata record (a Python ``list[float]``
  serialises cleanly to MongoDB, no separate vector index needed).
- Similarity is plain cosine over Python lists. At 100 chapters × ~768 dims
  this is well under a millisecond and avoids any vector-DB dependency.
- All entry points fail soft: a retrieval/embedding error never blocks the
  writer. The caller falls back to the recency-only path.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from django.conf import settings
from langchain_together import TogetherEmbeddings

logger = logging.getLogger(__name__)

_embedder: Optional[TogetherEmbeddings] = None


def _get_embedder() -> Optional[TogetherEmbeddings]:
    """Lazy-init a single TogetherEmbeddings client.

    Returns ``None`` if the API key isn't configured — the caller treats
    that as "RAG disabled" rather than crashing.
    """
    global _embedder
    if _embedder is not None:
        return _embedder
    api_key = getattr(settings, "TOGETHER_API_KEY", "") or ""
    if not api_key:
        logger.warning("RAG disabled — TOGETHER_API_KEY is not set")
        return None
    model = getattr(settings, "TOGETHER_EMBEDDING_MODEL", "") or ""
    if not model:
        logger.warning("RAG disabled — TOGETHER_EMBEDDING_MODEL is not set")
        return None
    _embedder = TogetherEmbeddings(api_key=api_key, model=model)
    return _embedder


async def embed_text(text: str) -> Optional[List[float]]:
    """Embed a single string. Returns ``None`` on any failure."""
    if not text or not text.strip():
        return None
    embedder = _get_embedder()
    if embedder is None:
        return None
    try:
        return await embedder.aembed_query(text)
    except Exception:
        logger.exception("Embedding call failed — RAG falls back to recency only")
        return None


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


async def select_relevant_summaries(
    query_text: str,
    chapter_metadata: List[Dict[str, Any]],
    *,
    k: int,
    exclude_recent: int,
) -> List[Dict[str, Any]]:
    """Return the top-K chapter_metadata entries most relevant to ``query_text``.

    The last ``exclude_recent`` chapters are skipped because the Writer
    already receives them as the recency anchor — duplicating them in the
    RAG section would waste context.

    Entries without a ``summary_embedding`` are skipped (legacy chapters
    written before RAG was enabled). Returns an empty list on any failure
    so the caller can fall back cleanly.
    """
    if k <= 0 or not chapter_metadata:
        return []
    candidate_pool = (
        chapter_metadata[:-exclude_recent]
        if exclude_recent > 0
        else list(chapter_metadata)
    )
    candidates = [m for m in candidate_pool if m.get("summary_embedding")]
    if not candidates:
        return []

    query_embedding = await embed_text(query_text)
    if query_embedding is None:
        return []

    scored = [
        (_cosine(query_embedding, m["summary_embedding"]), m)
        for m in candidates
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [m for _, m in scored[:k]]


def build_retrieval_query(chapter_plan: Dict[str, Any]) -> str:
    """Compose a retrieval query from the upcoming chapter's plan.

    Uses title, summary, key events, and characters — the fields that best
    describe what the next chapter will be about.
    """
    parts: List[str] = []
    title = chapter_plan.get("title") or ""
    summary = chapter_plan.get("summary") or ""
    key_events = chapter_plan.get("key_events") or []
    characters = chapter_plan.get("characters_involved") or []
    if title:
        parts.append(title)
    if summary:
        parts.append(summary)
    if key_events:
        parts.append("Events: " + "; ".join(str(e) for e in key_events))
    if characters:
        parts.append("Characters: " + ", ".join(str(c) for c in characters))
    return ". ".join(parts)
