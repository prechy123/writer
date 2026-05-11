"""DRF views for /api/v2/.

Phase 1: health + provider probe.
Phase 2: profile generation, Quick/Deep story creation, paste-parse,
         bible CRUD.
Phase 8: /start/, /continue/.
Phase 9: SSE streaming, reader-facing detail (characters/chapters/conclusion),
         full skeleton/ tree.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from asgiref.sync import async_to_sync
from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .streaming import format_sse, get_event_bus


# ---------------------------------------------------------------------------
# Dashboard page (single-page UI)
# ---------------------------------------------------------------------------

def dashboard(request):
    """Render the v2 single-page UI. Hash-based routes inside the page
    cover profiles, story creation (Quick / Deep / Paste), bible review,
    live event streaming, and the Skeleton inspect panel."""
    return render(request, "stories_v2/dashboard.html")

from . import mongo
from .agents import (
    build_profile,
    build_world_bible,
    forge_cast,
    parse_pasted_notes,
)
from .engine import (
    apply_manual_edit,
    cascade_regenerate_from,
    launch_story_run,
    regenerate_single_scene,
)
from .providers import get_router
from .schemas_v2 import (
    CharacterBibleV2,
    CharacterTier,
    DeepCharacter,
    DeepSurvey,
    DeepWorld,
    ProfileV2,
    ProfileV2Input,
    QuickSurvey,
    StoryStatus,
    character_budget_for,
)
from .schemas_v2.survey import PastedNotes
from .serializers import (
    CharacterBiblePatchSerializer,
    DeepSurveySerializer,
    PastedNotesSerializer,
    ProfileGenerateSerializer,
    QuickSurveySerializer,
    WorldBiblePatchSerializer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health + smoke
# ---------------------------------------------------------------------------

@api_view(["GET"])
def health(request):
    db = mongo.get_db()
    try:
        collections = sorted(db.list_collection_names())
    except Exception as exc:
        return Response(
            {"ok": False, "error": f"mongo unreachable: {exc}"},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    router = get_router()
    return Response(
        {
            "ok": True,
            "version": "v2",
            "atlas_detected": mongo.is_atlas(),
            "collections_present": [
                c
                for c in [
                    mongo.COL_PROFILES,
                    mongo.COL_STORIES,
                    mongo.COL_CHARACTERS,
                    mongo.COL_WORLDS,
                    mongo.COL_BEATS,
                    mongo.COL_SCENES,
                    mongo.COL_EVENTS,
                ]
                if c in collections
            ],
            "providers_configured": router.available_providers(),
        }
    )


@api_view(["GET"])
def provider_probe(request):
    role = request.query_params.get("role", "critic_voice")
    router = get_router()
    try:
        out = async_to_sync(router.chat_text)(
            role=role,
            system="You are a one-word echo.",
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            max_tokens=16,
            temperature=0.0,
        )
        return Response({"ok": True, "role": role, "reply": out.strip()})
    except KeyError as exc:
        return Response({"ok": False, "error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as exc:
        return Response(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )


# ---------------------------------------------------------------------------
# Profile v2
# ---------------------------------------------------------------------------

@api_view(["POST"])
def generate_profile_view(request):
    """POST /api/v2/profiles/  — create + persist a ProfileV2 from bio inputs."""
    serializer = ProfileGenerateSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    inputs: ProfileV2Input = serializer.pydantic
    profile_id = str(uuid.uuid4())

    try:
        payload = async_to_sync(build_profile)(inputs)
    except Exception as exc:
        logger.exception("profile generation failed")
        return Response(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    profile = ProfileV2(
        profile_id=profile_id,
        name=inputs.name,
        inputs=inputs,
        lexical_fingerprint=payload["lexical_fingerprint"],
        emotional_defaults=payload["emotional_defaults"],
        preferred_phrases=payload["preferred_phrases"],
        banned_phrases=payload["banned_phrases"],
        few_shot_samples=payload["few_shot_samples"],
        expertise_tags=payload["expertise_tags"],
    )
    doc = {"_id": profile_id, **profile.model_dump()}
    mongo.insert_profile(doc)
    return Response(
        {
            "ok": True,
            "profile_id": profile_id,
            "name": profile.name,
            "preview": {
                "lexical_fingerprint": profile.lexical_fingerprint.model_dump(),
                "emotional_defaults": profile.emotional_defaults.model_dump(),
                "preferred_phrases": profile.preferred_phrases,
                "banned_phrases": profile.banned_phrases,
                "few_shot_count": len(profile.few_shot_samples),
            },
        },
        status=status.HTTP_201_CREATED,
    )


@api_view(["GET"])
def list_profiles_view(request):
    return Response({"profiles": [_clean(p) for p in mongo.list_profiles()]})


@api_view(["GET"])
def get_profile_view(request, profile_id: str):
    doc = mongo.get_profile(profile_id)
    if not doc:
        return Response({"detail": "Profile not found."}, status=status.HTTP_404_NOT_FOUND)
    return Response(_clean(doc))


# ---------------------------------------------------------------------------
# Paste-anything parser
# ---------------------------------------------------------------------------

@api_view(["POST"])
def parse_pasted_view(request):
    """POST /api/v2/stories/parse/  — pre-fill the Deep wizard from a paste."""
    serializer = PastedNotesSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    notes: PastedNotes = serializer.pydantic
    try:
        parsed = async_to_sync(parse_pasted_notes)(notes)
    except Exception as exc:
        logger.exception("paste parser failed")
        return Response(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )
    return Response({"ok": True, "draft": parsed.model_dump()})


# ---------------------------------------------------------------------------
# Story creation — Phase 2 stops at "envelope + bibles persisted"
# Generation kicks in once Phase 8 (orchestrator) lands.
# ---------------------------------------------------------------------------

def _build_initial_envelope(*, story_id: str, quick: QuickSurvey, deep: DeepSurvey | None) -> Dict[str, Any]:
    now = datetime.datetime.utcnow()
    progress = {
        "stage": "queued",
        "message": "Story queued. Bibles built; awaiting generation engine.",
        "percent": 0,
        "completed_chapters": 0,
        "total_chapters": quick.num_chapters,
        "updated_at": now,
    }
    return {
        "_id": story_id,
        "title": quick.title,
        "status": StoryStatus.PENDING.value,
        "quick_survey": quick.model_dump(),
        "deep_survey": deep.model_dump() if deep else None,
        "arc_plan": None,
        "chapters": [],
        "current_chapter_idx": 0,
        "current_scene_idx": 0,
        "progress": progress,
        "progress_log": [progress],
        "created_at": now,
        "updated_at": now,
        "hidden": False,
        "character_budget": character_budget_for(quick.num_chapters),
    }


def _persist_bibles(
    *,
    story_id: str,
    quick: QuickSurvey,
    deep: DeepSurvey | None,
) -> Dict[str, Any]:
    """Build + persist world bible + cast. Returns a summary dict."""
    # World
    user_world: DeepWorld | None = deep.world if deep else None
    world = async_to_sync(build_world_bible)(
        story_id=story_id,
        title=quick.title,
        premise=quick.premise,
        genres=quick.genres,
        tone=quick.tone,
        user_world=user_world,
    )
    mongo.upsert_world_bible(story_id, world.model_dump())

    # Cast
    user_characters: List[DeepCharacter] = []
    if deep and deep.characters:
        user_characters = list(deep.characters)
    elif quick.characters:
        # Promote Quick character entries into DeepCharacter stubs.
        for qc in quick.characters:
            user_characters.append(
                DeepCharacter(
                    name=qc.name,
                    tier=CharacterTier.MAIN if qc.role.lower() in {"protagonist", "main", "lead"} else CharacterTier.RECURRING,
                    role=qc.role,
                    background=qc.trait,
                    sample_lines=[qc.sample_line] if qc.sample_line else [],
                )
            )

    cast = async_to_sync(forge_cast)(
        story_id=story_id,
        title=quick.title,
        premise=quick.premise,
        genres=quick.genres,
        tone=quick.tone,
        pov=quick.pov,
        num_chapters=quick.num_chapters,
        world_bible=world.model_dump(),
        user_characters=user_characters or None,
    )
    cast_docs = [
        {"_id": f"{story_id}:{c.character_id}", **c.model_dump()} for c in cast
    ]
    mongo.insert_character_bibles_bulk(cast_docs)

    tier_counts: Dict[str, int] = {"main": 0, "recurring": 0, "side": 0}
    for c in cast:
        tier_counts[c.tier.value] = tier_counts.get(c.tier.value, 0) + 1

    return {
        "world_bible_id": story_id,
        "cast_size": len(cast),
        "tier_counts": tier_counts,
    }


@api_view(["POST"])
def create_story_quick(request):
    """POST /api/v2/stories/quick/  — Quick wizard story creation."""
    serializer = QuickSurveySerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    quick: QuickSurvey = serializer.pydantic
    story_id = str(uuid.uuid4())
    envelope = _build_initial_envelope(story_id=story_id, quick=quick, deep=None)
    mongo.insert_story_envelope(envelope)
    try:
        bibles = _persist_bibles(story_id=story_id, quick=quick, deep=None)
    except Exception as exc:
        logger.exception("quick: bible build failed")
        mongo.update_story_envelope(
            story_id,
            {"status": StoryStatus.FAILED.value, "progress": {"stage": "failed", "error": str(exc)}},
        )
        return Response(
            {"story_id": story_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )
    return Response(
        {
            "ok": True,
            "story_id": story_id,
            "status": StoryStatus.PENDING.value,
            "bibles": bibles,
            "note": "Bibles persisted. Generation engine wires in Phase 8 — call /api/v2/stories/<id>/start/ once available.",
        },
        status=status.HTTP_202_ACCEPTED,
    )


@api_view(["POST"])
def create_story_deep(request):
    """POST /api/v2/stories/deep/  — Deep wizard story creation."""
    serializer = DeepSurveySerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    deep: DeepSurvey = serializer.pydantic
    quick = deep.quick
    story_id = str(uuid.uuid4())
    envelope = _build_initial_envelope(story_id=story_id, quick=quick, deep=deep)
    mongo.insert_story_envelope(envelope)
    try:
        bibles = _persist_bibles(story_id=story_id, quick=quick, deep=deep)
    except Exception as exc:
        logger.exception("deep: bible build failed")
        mongo.update_story_envelope(
            story_id,
            {"status": StoryStatus.FAILED.value, "progress": {"stage": "failed", "error": str(exc)}},
        )
        return Response(
            {"story_id": story_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )
    return Response(
        {
            "ok": True,
            "story_id": story_id,
            "status": StoryStatus.PENDING.value,
            "bibles": bibles,
            "note": "Bibles persisted. Generation engine wires in Phase 8.",
        },
        status=status.HTTP_202_ACCEPTED,
    )


@api_view(["GET"])
def list_stories_view(request):
    """List stories with a slim summary (no full prose)."""
    docs = mongo.list_story_envelopes()
    summaries = []
    for d in docs:
        summaries.append({
            "story_id": d.get("_id"),
            "title": d.get("title"),
            "status": d.get("status"),
            "progress": d.get("progress"),
            "current_chapter_idx": d.get("current_chapter_idx"),
            "total_chapters": (d.get("quick_survey") or {}).get("num_chapters"),
            "created_at": d.get("created_at"),
            "updated_at": d.get("updated_at"),
        })
    return Response({"stories": summaries})


@api_view(["GET"])
def get_story_reader(request, story_id: str):
    """GET /api/v2/stories/<id>/

    Reader-facing detail. Only the three sections the user sees:
        characters (main + recurring only)
        chapters (with full prose)
        conclusion (when status == completed)

    Everything else (arc plan, world bible, beat sheets, scene drafts,
    mood vectors, critic findings) is served under /skeleton/.
    """
    story = mongo.get_story_envelope(story_id)
    if not story:
        return Response({"detail": "Story not found."}, status=status.HTTP_404_NOT_FOUND)

    # Cast — main + recurring only, with the reader-safe projection.
    cast_docs = mongo.list_character_bibles(story_id)
    visible_cast = [
        {
            "character_id": c.get("character_id"),
            "name": c.get("name"),
            "role": c.get("role"),
            "tier": c.get("tier"),
            "short_description": c.get("short_description"),
            "portrait_blurb": c.get("portrait_blurb"),
        }
        for c in cast_docs
        if (c.get("tier") in {"main", "recurring"}) and c.get("name")
    ]

    # Chapters — already stored on the story envelope as a list of dicts
    chapters = []
    for ch in (story.get("chapters") or []):
        chapters.append({
            "chapter_number": ch.get("chapter_number") or (ch.get("chapter_idx", 0) + 1),
            "title": ch.get("title"),
            "summary": ch.get("summary"),
            "text": ch.get("text"),
            "word_count": ch.get("word_count"),
            "committed_at": ch.get("committed_at"),
        })

    # Conclusion — populated when status == completed
    conclusion: Optional[Dict[str, Any]] = None
    if story.get("status") == "completed" and chapters:
        last = chapters[-1]
        ledger = story.get("continuity_ledger") or {}
        conclusion = {
            "closing_chapter_title": last.get("title"),
            "closing_excerpt": _trim_excerpt(last.get("text") or ""),
            "final_word_count": sum(int(c.get("word_count") or 0) for c in chapters),
            "open_threads_left": ledger.get("open_threads") or [],
            "unresolved_cliffhangers": ledger.get("unresolved_cliffhangers") or [],
        }

    return Response({
        "story_id": story_id,
        "title": story.get("title"),
        "status": story.get("status"),
        "progress": story.get("progress"),
        "created_at": story.get("created_at"),
        "updated_at": story.get("updated_at"),
        "hidden": bool(story.get("hidden")),
        "characters": visible_cast,
        "chapters": chapters,
        "conclusion": conclusion,
    })


# ---------------------------------------------------------------------------
# Skeleton (engine internals — behind a "Skeleton" button on the frontend)
# ---------------------------------------------------------------------------

@api_view(["POST"])
def regenerate_scene_view(request, story_id: str):
    """POST /api/v2/stories/<id>/regenerate-scene/

    Body: { "chapter_idx": int, "scene_idx": int, "user_notes": "<optional>" }

    Surgical regeneration — only the named scene is redrafted. Downstream
    state (mood histories, ledger entries from later scenes) is left
    alone. For full cascade replay, use /regenerate-from/.
    """
    data = request.data if isinstance(request.data, dict) else {}
    try:
        chapter_idx = int(data["chapter_idx"])
        scene_idx = int(data["scene_idx"])
    except (KeyError, TypeError, ValueError):
        return Response(
            {"detail": "chapter_idx and scene_idx are required integers."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    user_notes = data.get("user_notes")

    try:
        scene_doc = async_to_sync(regenerate_single_scene)(
            story_id=story_id,
            chapter_idx=chapter_idx,
            scene_idx=scene_idx,
            user_notes=user_notes,
        )
    except LookupError as exc:
        return Response({"detail": str(exc)}, status=status.HTTP_404_NOT_FOUND)
    except Exception as exc:
        logger.exception("regenerate-scene failed")
        return Response(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )
    return Response({
        "ok": True,
        "story_id": story_id,
        "chapter_idx": chapter_idx,
        "scene_idx": scene_idx,
        "word_count": scene_doc.get("word_count"),
    })


@api_view(["PATCH"])
def patch_scene_view(request, story_id: str, chapter_idx: int, scene_idx: int):
    """PATCH /api/v2/stories/<id>/scenes/<chapter>/<scene>/

    Body: { "prose": "...", "run_humaniser": true }
    """
    data = request.data if isinstance(request.data, dict) else {}
    prose = data.get("prose")
    if not prose or not str(prose).strip():
        return Response(
            {"detail": "'prose' is required."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    run_humaniser = bool(data.get("run_humaniser", True))

    try:
        scene_doc = async_to_sync(apply_manual_edit)(
            story_id=story_id,
            chapter_idx=int(chapter_idx),
            scene_idx=int(scene_idx),
            new_prose=str(prose),
            run_humaniser=run_humaniser,
        )
    except LookupError as exc:
        return Response({"detail": str(exc)}, status=status.HTTP_404_NOT_FOUND)
    except ValueError as exc:
        return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as exc:
        logger.exception("patch-scene failed")
        return Response(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )
    return Response({
        "ok": True,
        "story_id": story_id,
        "chapter_idx": chapter_idx,
        "scene_idx": scene_idx,
        "word_count": scene_doc.get("word_count"),
        "humanisation": scene_doc.get("humanisation_report"),
    })


@api_view(["POST"])
def regenerate_from_view(request, story_id: str):
    """POST /api/v2/stories/<id>/regenerate-from/

    Body: { "chapter_idx": int, "scene_idx": int }

    Wipes state from (chapter_idx, scene_idx) forward and re-launches
    the orchestrator. Trims mood histories, clears open threads /
    cliffhangers / plot seed events from after the cutoff. The arc plan
    is preserved.
    """
    data = request.data if isinstance(request.data, dict) else {}
    try:
        chapter_idx = int(data["chapter_idx"])
    except (KeyError, TypeError, ValueError):
        return Response(
            {"detail": "chapter_idx is required."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    scene_idx = data.get("scene_idx", 0)
    try:
        scene_idx = int(scene_idx)
    except (TypeError, ValueError):
        scene_idx = 0

    try:
        result = cascade_regenerate_from(
            story_id=story_id,
            from_chapter_idx=chapter_idx,
            from_scene_idx=scene_idx,
        )
    except LookupError as exc:
        return Response({"detail": str(exc)}, status=status.HTTP_404_NOT_FOUND)
    except Exception as exc:
        logger.exception("regenerate-from failed")
        return Response(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )
    return Response(result, status=status.HTTP_202_ACCEPTED)


@api_view(["POST"])
def hide_story_view(request, story_id: str):
    updated = mongo.set_story_hidden(story_id, True)
    if not updated:
        return Response({"detail": "Story not found."}, status=status.HTTP_404_NOT_FOUND)
    return Response({"ok": True, "story_id": story_id, "hidden": True})


@api_view(["POST"])
def unhide_story_view(request, story_id: str):
    updated = mongo.set_story_hidden(story_id, False)
    if not updated:
        return Response({"detail": "Story not found."}, status=status.HTTP_404_NOT_FOUND)
    return Response({"ok": True, "story_id": story_id, "hidden": False})


@api_view(["POST"])
def hide_profile_view(request, profile_id: str):
    updated = mongo.set_profile_hidden(profile_id, True)
    if not updated:
        return Response({"detail": "Profile not found."}, status=status.HTTP_404_NOT_FOUND)
    return Response({"ok": True, "profile_id": profile_id, "hidden": True})


@api_view(["POST"])
def unhide_profile_view(request, profile_id: str):
    updated = mongo.set_profile_hidden(profile_id, False)
    if not updated:
        return Response({"detail": "Profile not found."}, status=status.HTTP_404_NOT_FOUND)
    return Response({"ok": True, "profile_id": profile_id, "hidden": False})


@api_view(["GET"])
def skeleton_view(request, story_id: str):
    """GET /api/v2/stories/<id>/skeleton/  — the full engine state."""
    story = mongo.get_story_envelope(story_id)
    if not story:
        return Response({"detail": "Story not found."}, status=status.HTTP_404_NOT_FOUND)

    cast = mongo.list_character_bibles(story_id)
    world = mongo.col(mongo.COL_WORLDS).find_one({"story_id": story_id}, {"_id": 0})
    beats = list(mongo.col(mongo.COL_BEATS).find({"story_id": story_id}, {"_id": 0}).sort("chapter_idx", 1))
    scenes = list(
        mongo.col(mongo.COL_SCENES)
        .find({"story_id": story_id}, {"_id": 0, "embedding": 0})
        .sort([("chapter_idx", 1), ("scene_idx", 1)])
    )

    return Response({
        "story_id": story_id,
        "arc_plan": story.get("arc_plan"),
        "world_bible": world,
        "character_bibles": [_strip_mongo_id(c) for c in cast],
        "beat_sheets": beats,
        "scene_drafts": scenes,
        "continuity_ledger": story.get("continuity_ledger"),
        "quick_survey": story.get("quick_survey"),
        "deep_survey": story.get("deep_survey"),
        "character_budget": story.get("character_budget"),
        "progress_log_recent": (story.get("progress_log") or [])[-20:],
    })


@api_view(["GET"])
def skeleton_arc_view(request, story_id: str):
    story = mongo.get_story_envelope(story_id)
    if not story:
        return Response({"detail": "Story not found."}, status=status.HTTP_404_NOT_FOUND)
    return Response({"arc_plan": story.get("arc_plan")})


@api_view(["GET"])
def skeleton_world_view(request, story_id: str):
    world = mongo.col(mongo.COL_WORLDS).find_one({"story_id": story_id}, {"_id": 0})
    if not world:
        return Response({"detail": "World bible not found."}, status=status.HTTP_404_NOT_FOUND)
    return Response(world)


@api_view(["GET"])
def skeleton_characters_view(request, story_id: str):
    cast = mongo.list_character_bibles(story_id)
    return Response({"characters": [_strip_mongo_id(c) for c in cast]})


@api_view(["GET"])
def skeleton_chapter_beats_view(request, story_id: str, chapter_idx: int):
    beat = mongo.col(mongo.COL_BEATS).find_one(
        {"story_id": story_id, "chapter_idx": int(chapter_idx)}, {"_id": 0}
    )
    if not beat:
        return Response({"detail": "Beat sheet not found for that chapter."}, status=status.HTTP_404_NOT_FOUND)
    return Response(beat)


@api_view(["GET"])
def skeleton_chapter_scenes_view(request, story_id: str, chapter_idx: int):
    scenes = list(
        mongo.col(mongo.COL_SCENES)
        .find({"story_id": story_id, "chapter_idx": int(chapter_idx)}, {"_id": 0, "embedding": 0})
        .sort("scene_idx", 1)
    )
    return Response({"scenes": scenes})


@api_view(["GET"])
def skeleton_events_view(request, story_id: str):
    """GET /api/v2/stories/<id>/skeleton/events/?after_seq=N

    Replay run events for a story. Used by the frontend's debug/inspect
    panel and by clients reconnecting to the SSE stream.
    """
    try:
        after_seq = int(request.query_params.get("after_seq", "0"))
    except (TypeError, ValueError):
        after_seq = 0
    events = mongo.fetch_run_events_after(story_id, after_seq=after_seq, limit=500)
    out = []
    for e in events:
        out.append({
            "seq": e.get("seq"),
            "event_type": e.get("event_type"),
            "payload": e.get("payload"),
            "created_at": e.get("created_at"),
        })
    return Response({"events": out, "after_seq": after_seq})


# ---------------------------------------------------------------------------
# SSE — async streaming endpoint
# ---------------------------------------------------------------------------

@csrf_exempt
@require_http_methods(["GET"])
def stream_story_events(request, story_id: str):
    """GET /api/v2/stories/<id>/stream/?after_seq=N

    Server-Sent Events stream. Sends:
        - heartbeats every 15s
        - per-event SSE frames (id, event, data) for everything published
          to the story event bus.

    Clients can pass ``?after_seq=N`` to replay missed events before the
    live tail begins.
    """
    try:
        after_seq = int(request.GET.get("after_seq", "0"))
    except (TypeError, ValueError):
        after_seq = 0

    response = StreamingHttpResponse(
        _sse_stream(story_id, after_seq=after_seq),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    response["Connection"] = "keep-alive"
    return response


async def _sse_stream(story_id: str, *, after_seq: int) -> AsyncIterator[bytes]:
    """Async generator yielding SSE-formatted bytes."""
    bus = get_event_bus()
    # Initial comment line so proxies don't time out the response.
    yield b": connected\n\n"

    async def _publisher():
        async for event in bus.subscribe(story_id, after_seq=after_seq):
            yield format_sse(event).encode("utf-8")
            if event.get("event_type") in {"story.completed", "story.failed"}:
                yield b": stream-end\n\n"
                break

    # We interleave a 15s heartbeat with the event stream by racing the
    # publisher against a sleep coroutine. asyncio.wait makes that clean.
    pub = _publisher().__aiter__()
    heartbeat_interval = 15.0
    next_event_task: Optional[asyncio.Task] = None
    while True:
        if next_event_task is None:
            next_event_task = asyncio.create_task(pub.__anext__())
        heartbeat_task = asyncio.create_task(asyncio.sleep(heartbeat_interval))
        done, _ = await asyncio.wait(
            {next_event_task, heartbeat_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if next_event_task in done:
            try:
                chunk = next_event_task.result()
                yield chunk
                next_event_task = None
            except StopAsyncIteration:
                heartbeat_task.cancel()
                break
            except Exception as exc:
                logger.exception("sse: publisher error for %s", story_id)
                heartbeat_task.cancel()
                yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n".encode("utf-8")
                break
            heartbeat_task.cancel()
        else:
            # heartbeat fired
            yield b": heartbeat\n\n"


@api_view(["POST"])
def start_story_view(request, story_id: str):
    """POST /api/v2/stories/<story_id>/start/

    Kicks off the generation engine for the first batch. The Quick/Deep
    wizard endpoints persist envelope + bibles synchronously, then the
    frontend calls this to start writing.
    """
    story = mongo.get_story_envelope(story_id)
    if not story:
        return Response({"detail": "Story not found."}, status=status.HTTP_404_NOT_FOUND)

    current_status = story.get("status")
    if current_status in {"running", "writing", "architecting", "planning"}:
        return Response(
            {"detail": f"Story is already {current_status}."},
            status=status.HTTP_409_CONFLICT,
        )
    if current_status == "completed":
        return Response(
            {"detail": "Story is already completed."},
            status=status.HTTP_409_CONFLICT,
        )

    batch_size = request.data.get("batch_size") if isinstance(request.data, dict) else None
    try:
        batch_size_int = int(batch_size) if batch_size is not None else None
    except (TypeError, ValueError):
        batch_size_int = None

    launch_story_run(story_id, batch_size=batch_size_int)
    return Response(
        {
            "ok": True,
            "story_id": story_id,
            "status": "queued",
            "note": "Generation started in background. Poll GET /api/v2/stories/<id>/ "
                    "or subscribe to /stream/ for events.",
        },
        status=status.HTTP_202_ACCEPTED,
    )


@api_view(["POST"])
def continue_story_view(request, story_id: str):
    """POST /api/v2/stories/<story_id>/continue/

    Resumes generation from current_chapter_idx for one more batch.
    """
    story = mongo.get_story_envelope(story_id)
    if not story:
        return Response({"detail": "Story not found."}, status=status.HTTP_404_NOT_FOUND)

    current_status = story.get("status")
    if current_status not in {"awaiting_continue", "failed"}:
        return Response(
            {"detail": f"Story status is {current_status!r}; must be 'awaiting_continue' to continue."},
            status=status.HTTP_409_CONFLICT,
        )

    total = int(story.get("quick_survey", {}).get("num_chapters", 0) or 0)
    starting = int(story.get("current_chapter_idx") or 0)
    if starting >= total:
        return Response(
            {"detail": "All chapters already written."},
            status=status.HTTP_409_CONFLICT,
        )

    additional = (request.data or {}).get("additional_chapters")
    try:
        batch = int(additional) if additional is not None else None
    except (TypeError, ValueError):
        batch = None
    if batch is None:
        batch = min(3, total - starting)
    batch = max(1, min(batch, total - starting))

    launch_story_run(story_id, batch_size=batch)
    return Response(
        {
            "ok": True,
            "story_id": story_id,
            "status": "queued",
            "batch_size": batch,
            "starting_chapter_idx": starting,
        },
        status=status.HTTP_202_ACCEPTED,
    )


# ---------------------------------------------------------------------------
# Bible CRUD
# ---------------------------------------------------------------------------

@api_view(["GET"])
def get_character_view(request, story_id: str, character_id: str):
    doc = mongo.get_character_bible(story_id, character_id)
    if not doc:
        return Response({"detail": "Character not found."}, status=status.HTTP_404_NOT_FOUND)
    return Response(_clean(doc))


@api_view(["PATCH"])
def patch_character_view(request, story_id: str, character_id: str):
    serializer = CharacterBiblePatchSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    patch: Dict[str, Any] = dict(serializer.validated_data)

    existing = mongo.get_character_bible(story_id, character_id)
    if not existing:
        return Response({"detail": "Character not found."}, status=status.HTTP_404_NOT_FOUND)

    # Strip immutable fields.
    patch.pop("character_id", None)
    patch.pop("story_id", None)
    patch.pop("created_at", None)

    merged = {**existing, **patch}
    try:
        validated = CharacterBibleV2.model_validate(merged)
    except Exception as exc:
        return Response(
            {"detail": "Validation failed.", "error": str(exc)},
            status=status.HTTP_400_BAD_REQUEST,
        )
    updated = mongo.update_character_bible(
        story_id, character_id, validated.model_dump(exclude={"character_id", "story_id", "created_at"})
    )
    return Response(_clean(updated or {}))


@api_view(["GET"])
def list_characters_view(request, story_id: str):
    tier = request.query_params.get("tier")
    docs = mongo.list_character_bibles(story_id, tier=tier)
    return Response({"characters": [_clean(d) for d in docs]})


@api_view(["GET"])
def get_world_view(request, story_id: str):
    doc = mongo.get_world_bible(story_id)
    if not doc:
        return Response({"detail": "World bible not found."}, status=status.HTTP_404_NOT_FOUND)
    return Response(_clean(doc))


@api_view(["PATCH"])
def patch_world_view(request, story_id: str):
    serializer = WorldBiblePatchSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    patch: Dict[str, Any] = dict(serializer.validated_data)

    existing = mongo.get_world_bible(story_id)
    if not existing:
        return Response({"detail": "World bible not found."}, status=status.HTTP_404_NOT_FOUND)
    patch.pop("story_id", None)
    patch.pop("created_at", None)
    merged = {**existing, **patch}

    # Validate the merged document against WorldBibleV2.
    from .schemas_v2 import WorldBibleV2

    try:
        validated = WorldBibleV2.model_validate(merged)
    except Exception as exc:
        return Response(
            {"detail": "Validation failed.", "error": str(exc)},
            status=status.HTTP_400_BAD_REQUEST,
        )
    updated = mongo.update_world_bible(
        story_id, validated.model_dump(exclude={"story_id", "created_at"})
    )
    return Response(_clean(updated or {}))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Mongo ObjectId / datetime → JSON-safe.  Strip the internal ``_id``
    when it duplicates a string key like ``profile_id`` / ``character_id``."""
    if not doc:
        return {}
    out: Dict[str, Any] = {}
    for k, v in doc.items():
        if k == "_id":
            # Keep a string version under id_; main string id is elsewhere.
            out["mongo_id"] = str(v)
            continue
        out[k] = v
    return out


def _strip_mongo_id(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in doc.items() if k != "_id"}


def _trim_excerpt(text: str, *, words: int = 200) -> str:
    parts = (text or "").split()
    if len(parts) <= words:
        return text
    return " ".join(parts[-words:])
