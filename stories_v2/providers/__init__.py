"""Multi-provider LLM client layer for stories_v2.

Free-tier first: Groq, Gemini, OpenRouter, xAI (Grok), and the legacy
Together AI client are all exposed through a single ``ChatClient``
protocol (``providers/base.py``). The ``router`` selects per-role
preferences from ``policy`` and falls back across providers on
rate-limit or server errors.

Public API:
    get_router()           -> Router singleton
    Router.chat_json(...)  -> dict (validated against optional Pydantic model)
    Router.chat_text(...)  -> str  (free-form completion)
    Router.chat_stream(...) -> async iterator of str chunks
    Router.embed(text)     -> list[float] | None
"""

from .router import Router, get_router

__all__ = ["Router", "get_router"]
