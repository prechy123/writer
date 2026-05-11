"""Role-aware provider router with retry + fallback.

The router holds one instance of each provider client and consults
``policy.DEFAULT_POLICY`` (overridable per-call) to pick which provider
to try first for a given role. On rate-limit or transient failure it
rotates to the next provider; on a hard config error (no API key) it
silently skips that provider.

Locked to two providers: Groq + Together AI. Both expose the OpenAI
Chat Completions wire format, so they share ``OpenAICompatClient``.

Usage from an agent:

    from stories_v2.providers import get_router
    router = get_router()
    plan = await router.chat_json(
        role="architect",
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        schema=ArcPlan,
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any, AsyncIterator, Dict, List, Optional, Type, TypeVar

from django.conf import settings
from pydantic import BaseModel

from .base import (
    ChatClient,
    ConfigError,
    ProviderError,
    RateLimitError,
    TransientError,
)
from .openai_compat import OpenAICompatClient
from .policy import DEFAULT_POLICY, RouteSpec

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default) or getattr(settings, name, default) or ""


def _build_providers() -> Dict[str, ChatClient]:
    """Instantiate one client per supported provider.

    Only Groq + Together AI are supported. Missing API keys are tolerated
    — the client is still registered, but its first call will raise
    ``ConfigError`` and the router will skip to the next provider in
    the route.
    """
    return {
        "groq": OpenAICompatClient(
            name="groq",
            base_url=_env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=_env("GROQ_API_KEY"),
        ),
        "together": OpenAICompatClient(
            name="together",
            base_url=_env("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),
            api_key=_env("TOGETHER_API_KEY"),
        ),
    }


class Router:
    """Routes chat / embed calls across providers per role."""

    def __init__(
        self,
        *,
        providers: Optional[Dict[str, ChatClient]] = None,
        policy: Optional[Dict[str, List[RouteSpec]]] = None,
    ) -> None:
        self._providers = providers or _build_providers()
        self._policy = policy or DEFAULT_POLICY

    def _route(self, role: str) -> List[RouteSpec]:
        spec = self._policy.get(role)
        if not spec:
            raise KeyError(f"router: no policy for role {role!r}")
        return spec

    def available_providers(self) -> List[str]:
        """Return provider names that look configured (have an API key)."""
        configured: List[str] = []
        for name, client in self._providers.items():
            api_key = getattr(client, "_api_key", "")
            if api_key:
                configured.append(name)
        return configured

    async def chat_text(
        self,
        *,
        role: str,
        system: Optional[str],
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 120.0,
    ) -> str:
        last_err: Optional[Exception] = None
        for spec in self._route(role):
            client = self._providers.get(spec.provider)
            if client is None:
                continue
            try:
                return await client.chat_text(
                    model_id=spec.model_id,
                    system=system,
                    messages=messages,
                    temperature=temperature if temperature is not None else (spec.temperature or 0.7),
                    max_tokens=max_tokens if max_tokens is not None else spec.max_tokens,
                    timeout=timeout,
                )
            except (RateLimitError, TransientError, ConfigError) as exc:
                logger.warning("router[%s]: %s rotated (%s)", role, spec.provider, exc)
                last_err = exc
                await self._backoff(exc)
                continue
            except ProviderError as exc:
                logger.warning("router[%s]: %s hard error (%s)", role, spec.provider, exc)
                last_err = exc
                continue
        raise ProviderError(
            f"router[{role}]: all providers exhausted (last: {last_err!r})"
        ) from last_err

    async def chat_json(
        self,
        *,
        role: str,
        system: Optional[str],
        messages: List[Dict[str, str]],
        schema: Optional[Type[T]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 180.0,
    ) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for spec in self._route(role):
            client = self._providers.get(spec.provider)
            if client is None:
                continue
            try:
                return await client.chat_json(
                    model_id=spec.model_id,
                    system=system,
                    messages=messages,
                    schema=schema,
                    temperature=temperature if temperature is not None else (spec.temperature or 0.4),
                    max_tokens=max_tokens if max_tokens is not None else spec.max_tokens,
                    timeout=timeout,
                )
            except (RateLimitError, TransientError, ConfigError) as exc:
                logger.warning("router[%s]: %s rotated (%s)", role, spec.provider, exc)
                last_err = exc
                await self._backoff(exc)
                continue
            except ProviderError as exc:
                logger.warning("router[%s]: %s hard error (%s)", role, spec.provider, exc)
                last_err = exc
                continue
        raise ProviderError(
            f"router[{role}]: all providers exhausted (last: {last_err!r})"
        ) from last_err

    async def chat_stream(
        self,
        *,
        role: str,
        system: Optional[str],
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> AsyncIterator[str]:
        """Streams from the first available provider; does NOT fall back
        mid-stream (a partial token sequence can't be safely resumed on a
        different model). On total failure to start, raises ProviderError.
        """
        last_err: Optional[Exception] = None
        for spec in self._route(role):
            client = self._providers.get(spec.provider)
            if client is None:
                continue
            try:
                stream = client.chat_stream(
                    model_id=spec.model_id,
                    system=system,
                    messages=messages,
                    temperature=temperature if temperature is not None else (spec.temperature or 0.7),
                    max_tokens=max_tokens if max_tokens is not None else spec.max_tokens,
                    timeout=timeout,
                )
                async for chunk in stream:
                    yield chunk
                return
            except (RateLimitError, TransientError, ConfigError) as exc:
                logger.warning("router[%s]: %s stream rotated (%s)", role, spec.provider, exc)
                last_err = exc
                await self._backoff(exc)
                continue
        raise ProviderError(
            f"router[{role}]: stream — all providers exhausted (last: {last_err!r})"
        ) from last_err

    async def embed(self, text: str, *, role: str = "embed") -> Optional[List[float]]:
        for spec in self._route(role):
            client = self._providers.get(spec.provider)
            if client is None:
                continue
            try:
                vec = await client.embed(text, model_id=spec.model_id)
            except Exception:
                vec = None
            if vec:
                return vec
        return None

    async def _backoff(self, exc: Exception) -> None:
        # Rate-limit errors get a slightly longer pause; transients get a
        # short jittered pause. The router does not block long here —
        # the next provider gets tried immediately on average.
        if isinstance(exc, RateLimitError):
            await asyncio.sleep(0.8 + random.random() * 1.2)
        else:
            await asyncio.sleep(0.1 + random.random() * 0.3)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_router_instance: Optional[Router] = None


def get_router() -> Router:
    global _router_instance
    if _router_instance is None:
        _router_instance = Router()
    return _router_instance
