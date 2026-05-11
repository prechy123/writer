"""SSE event bus for v2 stories.

Each story has a per-process ``asyncio.Queue`` so the SSE endpoint can
stream events that the generation engine emits. Events are also
persisted to ``run_events_v2`` so reconnecting clients can replay
missed events via ``?after_seq=N``.

This is intentionally minimal in Phase 1 — Phase 9 expands the event
types and wires SSE into the DRF view layer.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional

from . import mongo

logger = logging.getLogger(__name__)


class StoryEventBus:
    """Per-story pub/sub. Subscribers get future events; replay comes
    from Mongo via ``mongo.fetch_run_events_after``."""

    def __init__(self) -> None:
        self._queues: Dict[str, list[asyncio.Queue]] = {}
        self._seq: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def publish(self, story_id: str, event_type: str, payload: Dict[str, Any]) -> int:
        async with self._lock:
            self._seq[story_id] = self._seq.get(story_id, 0) + 1
            seq = self._seq[story_id]

        try:
            mongo.append_run_event(story_id, seq, event_type, payload)
        except Exception:
            logger.exception("streaming: failed to persist event %s/%s", story_id, seq)

        event = {
            "story_id": story_id,
            "seq": seq,
            "event_type": event_type,
            "payload": payload,
            "ts": datetime.datetime.utcnow().isoformat(),
        }

        for q in list(self._queues.get(story_id, [])):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Slow consumer — drop and continue. Replay covers gaps.
                pass

        return seq

    async def subscribe(self, story_id: str, *, after_seq: int = 0) -> AsyncIterator[Dict[str, Any]]:
        """Yield events for one story. Starts with replay (after_seq+1..now)
        then yields live events as they arrive."""
        # Replay
        for past in mongo.fetch_run_events_after(story_id, after_seq=after_seq):
            yield {
                "story_id": story_id,
                "seq": past["seq"],
                "event_type": past["event_type"],
                "payload": past.get("payload") or {},
                "ts": past.get("created_at", datetime.datetime.utcnow()).isoformat()
                if isinstance(past.get("created_at"), datetime.datetime)
                else None,
            }

        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._queues.setdefault(story_id, []).append(q)
        try:
            while True:
                event = await q.get()
                yield event
                if event.get("event_type") in {"story.completed", "story.failed"}:
                    # Let the consumer decide to disconnect; but stop blocking
                    # if nothing else arrives within a reasonable window.
                    pass
        finally:
            try:
                self._queues.get(story_id, []).remove(q)
            except ValueError:
                pass


def format_sse(event: Dict[str, Any]) -> str:
    """Format an event for the wire (Server-Sent Events protocol)."""
    data = json.dumps(event, default=str)
    return f"id: {event.get('seq', 0)}\nevent: {event.get('event_type', 'message')}\ndata: {data}\n\n"


_bus: Optional[StoryEventBus] = None


def get_event_bus() -> StoryEventBus:
    global _bus
    if _bus is None:
        _bus = StoryEventBus()
    return _bus
