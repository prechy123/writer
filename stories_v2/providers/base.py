"""Provider-agnostic chat client protocol.

Every concrete provider (``openai_compat``, ``gemini``, ...) implements
this protocol. The router never touches a provider directly; it always
calls through the protocol so we can swap clients without touching the
agents.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ChatClient(Protocol):
    """All providers expose this surface.

    ``model_id`` is whatever the provider's API expects (e.g.
    ``llama-3.3-70b-versatile`` for Groq, ``gemini-2.5-flash`` for Gemini).
    """

    name: str  # short provider name, e.g. "groq", "gemini"

    async def chat_text(
        self,
        *,
        model_id: str,
        system: Optional[str],
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 90.0,
    ) -> str:
        """Free-form completion. Returns the assistant message content."""

    async def chat_json(
        self,
        *,
        model_id: str,
        system: Optional[str],
        messages: List[Dict[str, str]],
        schema: Optional[Type[T]] = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """JSON-mode completion.

        If ``schema`` is provided, the result is validated against it.
        Returns the parsed dict (or the validated model dumped via
        ``model_dump()``) so callers don't need to know which path was
        taken.
        """

    async def chat_stream(
        self,
        *,
        model_id: str,
        system: Optional[str],
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 180.0,
    ) -> AsyncIterator[str]:
        """Streamed completion. Yields successive token chunks."""

    async def embed(self, text: str, *, model_id: Optional[str] = None) -> Optional[List[float]]:
        """Embed a single string. Return ``None`` if embeddings aren't supported."""


class ProviderError(Exception):
    """Base class for routed provider errors."""


class RateLimitError(ProviderError):
    """Raised on 429 / quota responses so the router can back off + rotate."""


class TransientError(ProviderError):
    """Raised on 5xx, timeouts, connection resets — also triggers fallback."""


class ConfigError(ProviderError):
    """API key missing, model not enabled on this provider, etc."""
