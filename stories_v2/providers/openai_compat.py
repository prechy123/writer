"""OpenAI-compatible HTTP client shared by Groq, OpenRouter, xAI, and Together.

All four providers expose the OpenAI Chat Completions wire format at a
different base URL, so we drive them with a single ``httpx``-based
client. We avoid pulling the official ``openai`` SDK to keep the
dependency footprint small and to dodge the global env-var coupling
(``OPENAI_API_KEY``) the SDK insists on.

Each instance is configured for one provider:
    OpenAICompatClient(name="groq", base_url=..., api_key=...)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .base import ChatClient, ConfigError, ProviderError, RateLimitError, TransientError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAICompatClient(ChatClient):
    name: str

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        embed_path: Optional[str] = "/embeddings",
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._embed_path = embed_path
        self._client: Optional[httpx.AsyncClient] = None

    def _ensure(self) -> httpx.AsyncClient:
        if not self._api_key:
            raise ConfigError(f"{self.name}: API key is not configured")
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=30.0),
            )
        return self._client

    # -- helpers -------------------------------------------------------------

    def _build_messages(
        self,
        *,
        system: Optional[str],
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if system:
            out.append({"role": "system", "content": system})
        out.extend(messages)
        return out

    def _raise_from_response(self, resp: httpx.Response) -> None:
        if resp.status_code == 429:
            raise RateLimitError(f"{self.name}: 429 {resp.text[:200]}")
        if 500 <= resp.status_code < 600:
            raise TransientError(f"{self.name}: {resp.status_code} {resp.text[:200]}")
        if resp.status_code >= 400:
            raise ProviderError(f"{self.name}: {resp.status_code} {resp.text[:400]}")

    # -- public surface ------------------------------------------------------

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
        client = self._ensure()
        body = {
            "model": model_id,
            "messages": self._build_messages(system=system, messages=messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = await client.post("/chat/completions", json=body, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise TransientError(f"{self.name}: timeout") from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"{self.name}: {exc}") from exc
        self._raise_from_response(resp)
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.name}: malformed response {data!r}") from exc

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
        client = self._ensure()
        body: Dict[str, Any] = {
            "model": model_id,
            "messages": self._build_messages(system=system, messages=messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = await client.post("/chat/completions", json=body, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise TransientError(f"{self.name}: timeout") from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"{self.name}: {exc}") from exc
        self._raise_from_response(resp)
        data = resp.json()
        try:
            raw = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.name}: malformed response {data!r}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.name}: chat_json returned invalid JSON: {raw[:300]}") from exc

        if schema is not None:
            try:
                return schema.model_validate(parsed).model_dump()
            except ValidationError as exc:
                raise ProviderError(f"{self.name}: schema validation failed: {exc}") from exc
        return parsed

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
        client = self._ensure()
        body = {
            "model": model_id,
            "messages": self._build_messages(system=system, messages=messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            async with client.stream(
                "POST", "/chat/completions", json=body, timeout=timeout
            ) as resp:
                self._raise_from_response(resp)
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    delta = (
                        ((chunk.get("choices") or [{}])[0]).get("delta") or {}
                    ).get("content")
                    if delta:
                        yield delta
        except httpx.TimeoutException as exc:
            raise TransientError(f"{self.name}: timeout (stream)") from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"{self.name}: {exc}") from exc

    async def embed(self, text: str, *, model_id: Optional[str] = None) -> Optional[List[float]]:
        if not self._embed_path or not text or not text.strip():
            return None
        client = self._ensure()
        body = {"model": model_id, "input": text}
        try:
            resp = await client.post(self._embed_path, json=body, timeout=60.0)
        except httpx.HTTPError:
            return None
        if resp.status_code >= 400:
            return None
        data = resp.json()
        try:
            return data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError):
            return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["OpenAICompatClient"]
