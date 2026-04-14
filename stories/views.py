"""DRF views for story creation, polling, listing, and profile generation.

All views are synchronous to ensure compatibility with both ``runserver``
(WSGI) and ``uvicorn`` (ASGI).  Long-running async work (LLM calls, graph
execution) is bridged via ``async_to_sync`` or a background thread.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid

from asgiref.sync import async_to_sync
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .agents import run_profiling
from .graph import story_graph
from .mongodb import (
    create_story_record,
    get_profile,
    get_story,
    list_profiles,
    list_stories,
    save_generated_profile,
    update_story_status,
)
from .prompts import build_bio_context
from .serializers import (
    ProfileGenerateSerializer,
    StoryCreateSerializer,
    StoryDetailSerializer,
)
from .state import StoryState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background graph runner
# ---------------------------------------------------------------------------

async def _run_story_graph_async(story_id: str, initial_state: dict) -> None:
    """Execute the full LangGraph pipeline."""
    try:
        update_story_status(story_id, "running")
        logger.info("Story %s — graph execution started", story_id)

        result = await story_graph.ainvoke(initial_state)

        update_story_status(
            story_id,
            "completed",
            state=result,
            manuscript=result.get("final_manuscript"),
        )
        logger.info("Story %s — completed successfully", story_id)

    except Exception:
        logger.exception("Story %s — graph execution failed", story_id)
        update_story_status(story_id, "failed")


def _launch_story_graph(story_id: str, initial_state: dict) -> None:
    """Launch the graph in a background daemon thread with its own event loop.

    Works regardless of whether the caller is WSGI or ASGI.
    """
    def _target():
        asyncio.run(_run_story_graph_async(story_id, initial_state))

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    logger.info("Story %s — background thread started", story_id)


# ---------------------------------------------------------------------------
# Story API views
# ---------------------------------------------------------------------------

@api_view(["POST"])
def create_story(request):
    """POST /api/stories/create/

    Body: ``book_title``, ``book_description``, ``num_chapters``,
    and optionally ``profile_id``.

    - If ``profile_id`` is given → loads the stored profile and pre-populates
      author_profile / emotional_guidelines / expert_styles so Phase 1 agents
      are skipped entirely (zero extra LLM calls).
    - If ``profile_id`` is given but doesn't exist → returns 400.
    - If ``profile_id`` is omitted → Phase 1 agents run with generic prompts.
    """
    serializer = StoryCreateSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    story_id = str(uuid.uuid4())

    # Resolve profile
    author_profile = ""
    emotional_guidelines = ""
    expert_styles = ""
    profile_id = data.get("profile_id")

    if profile_id:
        profile = get_profile(profile_id)
        if profile is None:
            return Response(
                {"profile_id": f"Profile '{profile_id}' not found."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        author_profile = profile["author_profile"]
        emotional_guidelines = profile["emotional_guidelines"]
        expert_styles = profile["expert_styles"]
        logger.info("Story %s — using stored profile '%s'", story_id, profile_id)

    # Persist initial record
    create_story_record(
        story_id,
        data["book_title"],
        data["book_description"],
        data["num_chapters"],
    )

    # Build the seed state for LangGraph
    initial_state: StoryState = {
        "book_title": data["book_title"],
        "book_description": data["book_description"],
        "num_chapters": data["num_chapters"],
        "author_profile": author_profile,
        "emotional_guidelines": emotional_guidelines,
        "expert_styles": expert_styles,
        "story_plan": {},
        "chapters_to_write": [],
        "current_chapter_index": 0,
        "current_draft": "",
        "review_feedback": "",
        "retry_count": 0,
        "running_summary": "",
        "final_chapters": [],
        "final_manuscript": {},
        "status": "pending",
        "story_id": story_id,
        "error": "",
    }

    # Fire-and-forget in a background thread
    _launch_story_graph(story_id, initial_state)

    return Response(
        {
            "story_id": story_id,
            "status": "pending",
            "profile_used": bool(profile_id),
        },
        status=status.HTTP_202_ACCEPTED,
    )


@api_view(["GET"])
def get_story_detail(request, story_id: str):
    """GET /api/stories/<story_id>/"""
    doc = get_story(story_id)
    if doc is None:
        return Response(
            {"error": "Story not found"}, status=status.HTTP_404_NOT_FOUND
        )
    serializer = StoryDetailSerializer(doc)
    return Response(serializer.data)


@api_view(["GET"])
def list_stories_view(request):
    """GET /api/stories/"""
    docs = list_stories()
    results = [
        {
            "story_id": doc["_id"],
            "title": doc["title"],
            "status": doc["status"],
            "num_chapters": doc["num_chapters"],
            "created_at": doc["created_at"].isoformat(),
            "updated_at": doc["updated_at"].isoformat(),
        }
        for doc in docs
    ]
    return Response(results)


# ---------------------------------------------------------------------------
# Profile API views
# ---------------------------------------------------------------------------

@api_view(["POST"])
def generate_profile_view(request):
    """POST /api/profiles/generate/

    Accepts rich biographical context + writing samples, runs the 3 Phase 1
    agents (Profiler, Empath, Masterclass) and stores the generated outputs
    in MongoDB.  This is the one-time cost — all subsequent stories using
    this profile skip Phase 1 entirely.

    Note: this view blocks until profiling completes (typically 30-60s)
    because the user needs the profile_id before they can create stories.
    """
    serializer = ProfileGenerateSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    profile_id = str(uuid.uuid4())

    # Assemble biographical context from individual fields
    bio_context = build_bio_context(
        background=data.get("background", ""),
        personality=data.get("personality", ""),
        communication_style=data.get("communication_style", ""),
        interests_and_values=data.get("interests_and_values", ""),
        quirks=data.get("quirks", ""),
        additional_context=data.get("additional_context", ""),
    )

    writing_samples = data.get("writing_samples", [])
    expert_style = data.get("expert_style", "")
    expert_writing_sample = data.get("expert_writing_sample", "")

    # Run the 3 Phase 1 agents synchronously (blocks until done)
    results = async_to_sync(run_profiling)(
        name=data["name"],
        book_title=f"Profile for {data['name']}",
        book_description="General-purpose author voice profile.",
        bio_context=bio_context,
        writing_samples=writing_samples if writing_samples else None,
        expert_style=expert_style,
        expert_writing_sample=expert_writing_sample,
    )

    # Persist to MongoDB
    save_generated_profile(
        profile_id,
        data["name"],
        bio_context=bio_context,
        writing_samples=writing_samples,
        author_profile=results["author_profile"],
        emotional_guidelines=results["emotional_guidelines"],
        expert_styles=results["expert_styles"],
    )

    return Response(
        {
            "profile_id": profile_id,
            "name": data["name"],
            "author_profile_preview": results["author_profile"][:500] + "...",
            "emotional_guidelines_preview": results["emotional_guidelines"][:500] + "...",
            "expert_styles_preview": results["expert_styles"][:500] + "...",
        },
        status=status.HTTP_201_CREATED,
    )


@api_view(["GET"])
def get_profile_view(request, profile_id: str):
    """GET /api/profiles/<profile_id>/"""
    doc = get_profile(profile_id)
    if doc is None:
        return Response(
            {"error": "Profile not found"}, status=status.HTTP_404_NOT_FOUND
        )
    return Response(
        {
            "profile_id": doc["_id"],
            "name": doc["name"],
            "bio_context": doc.get("bio_context", ""),
            "samples_count": len(doc.get("writing_samples", [])),
            "author_profile": doc["author_profile"],
            "emotional_guidelines": doc["emotional_guidelines"],
            "expert_styles": doc["expert_styles"],
            "created_at": doc["created_at"].isoformat(),
        }
    )


@api_view(["GET"])
def list_profiles_view(request):
    """GET /api/profiles/"""
    docs = list_profiles()
    results = [
        {
            "profile_id": doc["_id"],
            "name": doc["name"],
            "created_at": doc["created_at"].isoformat(),
        }
        for doc in docs
    ]
    return Response(results)


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

def stories_dashboard(request):
    """Serve the single-page stories viewer."""
    return render(request, "stories/index.html")
