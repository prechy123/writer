"""3-tier memory architecture for long-form coherence.

Per scene generation, the Scene Writer assembles its context from:

    Working memory   — last 2 chapters verbatim (head + tail) + current
                       chapter's beat plan. Always present, never RAG'd.
    Episodic memory  — top-K scene summaries by vector similarity to the
                       upcoming scene's beat plan, excluding the working
                       window. This is the long-range continuity tier.
    Semantic memory  — character bibles for present_characters, active
                       world rules, open plot threads, unresolved
                       cliffhangers. Always injected.

Episodic retrieval has two implementations:
    atlas_vector.py  — Mongo Atlas $vectorSearch (server-side, scales).
    inmem_cosine.py  — Python-side cosine over scene_drafts_v2.embedding.
                       Used as a fallback when Atlas isn't detected.

``memory.retriever.assemble_context()`` is the public entry point.
"""

from .retriever import assemble_context, MemoryContext

__all__ = ["assemble_context", "MemoryContext"]
