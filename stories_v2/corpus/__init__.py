"""Style exemplar corpus.

A small bundled set of passages tagged by (genre, pov, pacing, emotion,
register, technique). The Scene Writer retrieves 2-3 passages per scene
that match the scene's profile and injects them as few-shot anchors.

The bundled set is intentionally small — it's the engine's reusable
"good writing" baseline. To extend, drop additional JSON files into
``corpus/exemplars/`` and re-run the bootstrap. Public-domain content
only (Project Gutenberg, government archives, original text).

Public API:
    list_exemplars(...)    -> list[CorpusEntry]
    pick_exemplars(...)    -> ordered top-K matches for a scene profile
"""

from .loader import list_exemplars, load_index
from .retriever import pick_exemplars
from .schema import CorpusEntry

__all__ = [
    "CorpusEntry",
    "list_exemplars",
    "load_index",
    "pick_exemplars",
]
