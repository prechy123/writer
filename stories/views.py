"""DRF views for story creation, polling, listing, and profile generation.

All views are synchronous to ensure compatibility with both ``runserver``
(WSGI) and ``uvicorn`` (ASGI).  Long-running async work (LLM calls, graph
execution) is bridged via ``async_to_sync`` or a background thread.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import threading
import uuid

from asgiref.sync import async_to_sync
from django.conf import settings as django_settings
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .agents import run_profiling
from .graph import continue_graph, story_graph
from .mongodb import (
    append_batch_log,
    create_story_record,
    get_and_lock_for_continue,
    get_profile,
    get_story,
    list_profiles,
    list_stories,
    release_continue_lock,
    save_generated_profile,
    set_story_hidden,
    update_story_status,
)
from .prompts import build_bio_context
from .serializers import (
    ProfileGenerateSerializer,
    StoryContinueSerializer,
    StoryCreateSerializer,
    StoryDetailSerializer,
)
from .state import StoryState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background graph runner
# ---------------------------------------------------------------------------

def _terminal_status(state: dict) -> str:
    """Decide a story's terminal status after a graph run.

    - ``completed``        — every planned chapter is written (publisher ran).
    - ``awaiting_continue`` — this batch hit its target but the book still
      has unwritten chapters; the /continue/ endpoint can resume it.
    """
    total = len(state.get("chapters_to_write") or [])
    current = int(state.get("current_chapter_index", 0) or 0)
    if total > 0 and current >= total and state.get("final_manuscript"):
        return "completed"
    return "awaiting_continue"


async def _run_graph_async(graph, story_id: str, initial_state: dict) -> None:
    """Execute a compiled LangGraph and persist the terminal status."""
    try:
        update_story_status(story_id, "running")
        logger.info("Story %s — graph execution started", story_id)

        result = await graph.ainvoke(initial_state)
        final_status = _terminal_status(result)

        # Close out the most recent batch_log entry if any.
        db_doc = get_story(story_id)
        batch_log = list(((db_doc or {}).get("batch_log")) or [])
        if batch_log and batch_log[-1].get("completed_at") is None:
            batch_log[-1] = {
                **batch_log[-1],
                "completed_at": datetime.datetime.utcnow(),
                "status": "completed" if final_status == "completed" else "batch_done",
                "final_chapter_index": int(result.get("current_chapter_index", 0) or 0),
            }

        update_story_status(
            story_id,
            final_status,
            state=result,
            manuscript=result.get("final_manuscript"),
            batch_log=batch_log or None,
        )
        logger.info(
            "Story %s — graph finished with status %s (chapters=%d/%d)",
            story_id,
            final_status,
            int(result.get("current_chapter_index", 0) or 0),
            len(result.get("chapters_to_write") or []),
        )

    except Exception:
        logger.exception("Story %s — graph execution failed", story_id)
        update_story_status(story_id, "failed")


def _launch_graph(graph, story_id: str, initial_state: dict) -> None:
    """Launch the graph in a background daemon thread with its own event loop.

    Works regardless of whether the caller is WSGI or ASGI.
    """
    def _target():
        asyncio.run(_run_graph_async(graph, story_id, initial_state))

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

    # Resolve the initial batch size: caller value takes priority, otherwise
    # the Django default. Clamp to [1, num_chapters].
    default_initial = getattr(django_settings, "DEFAULT_INITIAL_CHAPTERS", 3)
    requested_initial = data.get("initial_chapters") or default_initial
    initial_chapters = max(1, min(int(requested_initial), int(data["num_chapters"])))

    # Persist initial record
    create_story_record(
        story_id,
        data["book_title"],
        data["book_description"],
        data["num_chapters"],
        initial_chapters=initial_chapters,
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
        "target_chapter_index": initial_chapters,
        "chapter_metadata": [],
        "continuity_ledger": {},
        "batch_log": [],
        "final_manuscript": {},
        "status": "pending",
        "story_id": story_id,
        "error": "",
    }

    # Fire-and-forget in a background thread
    _launch_graph(story_graph, story_id, initial_state)

    return Response(
        {
            "story_id": story_id,
            "status": "pending",
            "profile_used": bool(profile_id),
            "initial_chapters": initial_chapters,
            "num_chapters": data["num_chapters"],
        },
        status=status.HTTP_202_ACCEPTED,
    )


@api_view(["POST"])
def continue_story_view(request, story_id: str):
    """POST /api/stories/<story_id>/continue/

    Body: ``{ "additional_chapters": int }``.

    Resumes a story that finished an initial batch and parked in status
    ``awaiting_continue``. Validates:

    - story exists and is currently ``awaiting_continue`` (else 404 / 409),
    - ``current_chapter_index + additional_chapters <= num_chapters``
      (else 400 — no over-requesting beyond the planned book).

    On success, launches the ``continue_graph`` which skips Phase 1-2 and
    resumes directly at the Writer, using the persisted story_plan,
    continuity_ledger, chapter_metadata, etc.
    """
    serializer = StoryContinueSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    additional = serializer.validated_data["additional_chapters"]

    # Quick existence check so 404 beats 409 for missing IDs.
    doc = get_story(story_id)
    if doc is None:
        return Response(
            {"error": "Story not found"}, status=status.HTTP_404_NOT_FOUND
        )

    # Atomic CAS lock: awaiting_continue → running.
    locked = get_and_lock_for_continue(story_id)
    if locked is None:
        return Response(
            {
                "error": (
                    f"Story is in status '{doc.get('status')}'. Continuation is "
                    "only allowed when status == 'awaiting_continue'."
                )
            },
            status=status.HTTP_409_CONFLICT,
        )

    state: dict = dict(locked.get("state") or {})
    total = len(state.get("chapters_to_write") or [])
    current = int(state.get("current_chapter_index", 0) or 0)
    remaining = total - current

    if remaining <= 0:
        release_continue_lock(story_id, restore_status="completed")
        return Response(
            {"error": "No chapters remain. The book is already complete."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if additional > remaining:
        release_continue_lock(story_id, restore_status="awaiting_continue")
        return Response(
            {
                "error": (
                    f"Requested {additional} chapters but only {remaining} remain "
                    f"(current={current}, total={total})."
                ),
                "chapters_remaining": remaining,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    new_target = current + additional
    state["target_chapter_index"] = new_target
    state["status"] = "running"
    # LangGraph needs these transient fields to be fresh for the next run.
    state["current_draft"] = ""
    state["review_feedback"] = ""
    state["retry_count"] = 0

    # Record a new batch_log entry before we kick off the run.
    batch_entry = {
        "start_idx": current,
        "end_idx": new_target,
        "started_at": datetime.datetime.utcnow(),
        "completed_at": None,
        "status": "running",
    }
    append_batch_log(story_id, batch_entry)

    _launch_graph(continue_graph, story_id, state)

    return Response(
        {
            "story_id": story_id,
            "status": "running",
            "current_chapter_index": current,
            "target_chapter_index": new_target,
            "chapters_remaining_after_batch": total - new_target,
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


@api_view(["POST"])
def hide_story_view(request, story_id: str):
    """POST /api/stories/<story_id>/hide/

    Soft-hides a story so it's excluded from ``GET /api/stories/``.
    The document itself is preserved and still reachable at
    ``GET /api/stories/<story_id>/``.
    """
    updated = set_story_hidden(story_id, True)
    if updated is None:
        return Response(
            {"error": "Story not found"}, status=status.HTTP_404_NOT_FOUND
        )
    return Response({"story_id": story_id, "hidden": True})


@api_view(["POST"])
def unhide_story_view(request, story_id: str):
    """POST /api/stories/<story_id>/unhide/

    Restores a hidden story to the default listing.
    """
    updated = set_story_hidden(story_id, False)
    if updated is None:
        return Response(
            {"error": "Story not found"}, status=status.HTTP_404_NOT_FOUND
        )
    return Response({"story_id": story_id, "hidden": False})


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
