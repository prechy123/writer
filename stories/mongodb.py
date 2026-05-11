"""PyMongo connection management and CRUD operations for story documents."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import certifi
from django.conf import settings
from pymongo import MongoClient, ReturnDocument

_client: Optional[MongoClient] = None


def get_db():
    """Return the MongoDB database handle (lazy-initialised singleton)."""
    global _client
    if _client is None:
        uri = settings.MONGODB_URI
        kwargs: Dict[str, Any] = {}
        if uri.startswith("mongodb+srv://") or "tls=true" in uri.lower():
            kwargs["tlsCAFile"] = certifi.where()
        _client = MongoClient(uri, **kwargs)
    return _client[settings.MONGODB_DB_NAME]


# ---------------------------------------------------------------------------
# Story CRUD
# ---------------------------------------------------------------------------

def create_story_record(
    story_id: str,
    title: str,
    description: str,
    num_chapters: int,
    *,
    initial_chapters: Optional[int] = None,
) -> Dict[str, Any]:
    """Insert an initial story document with status='pending'."""
    db = get_db()
    now = datetime.datetime.utcnow()
    progress = {
        "stage": "queued",
        "message": "Story generation queued.",
        "completed_chapters": 0,
        "total_chapters": int(num_chapters),
        "target_chapter_index": int(initial_chapters) if initial_chapters else int(num_chapters),
        "percent": 0,
        "updated_at": now,
    }
    batch_log: List[Dict[str, Any]] = []
    if initial_chapters:
        batch_log.append({
            "start_idx": 0,
            "end_idx": int(initial_chapters),
            "started_at": now,
            "completed_at": None,
            "status": "running",
        })
    doc = {
        "_id": story_id,
        "title": title,
        "description": description,
        "num_chapters": num_chapters,
        "initial_chapters": int(initial_chapters) if initial_chapters else num_chapters,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "state": {"progress": progress},
        "final_manuscript": None,
        "batch_log": batch_log,
        "progress": progress,
        "progress_log": [progress],
    }
    db.stories.insert_one(doc)
    return doc


# ---------------------------------------------------------------------------
# Continuation — atomic CAS lock
# ---------------------------------------------------------------------------

def get_and_lock_for_continue(story_id: str) -> Optional[Dict[str, Any]]:
    """Atomically move a story from ``awaiting_continue`` → ``running``.

    Returns the locked document (post-update) on success, or ``None`` if
    the story doesn't exist or is in the wrong status.  Using
    ``find_one_and_update`` makes this safe against concurrent
    ``/continue/`` requests.
    """
    db = get_db()
    return db.stories.find_one_and_update(
        {"_id": story_id, "status": "awaiting_continue"},
        {"$set": {
            "status": "running",
            "updated_at": datetime.datetime.utcnow(),
        }},
        return_document=ReturnDocument.AFTER,
    )


def release_continue_lock(story_id: str, restore_status: str = "awaiting_continue") -> None:
    """Release the CAS lock taken by ``get_and_lock_for_continue``.

    Used when validation fails after the lock has been acquired (e.g.
    the user asked for more chapters than remain).
    """
    db = get_db()
    db.stories.update_one(
        {"_id": story_id},
        {"$set": {
            "status": restore_status,
            "updated_at": datetime.datetime.utcnow(),
        }},
    )


def update_story_status(
    story_id: str,
    status: str,
    *,
    state: Optional[Dict[str, Any]] = None,
    manuscript: Optional[Dict[str, Any]] = None,
    batch_log: Optional[List[Dict[str, Any]]] = None,
    progress: Optional[Dict[str, Any]] = None,
) -> None:
    """Update the status (and optionally the full state / manuscript / batch log)."""
    db = get_db()
    if progress is not None:
        progress = {**progress, "updated_at": datetime.datetime.utcnow()}
        if state is not None:
            state = {**state, "progress": progress}
    update: Dict[str, Any] = {
        "$set": {
            "status": status,
            "updated_at": datetime.datetime.utcnow(),
        }
    }
    if state is not None:
        update["$set"]["state"] = state
    if manuscript is not None:
        update["$set"]["final_manuscript"] = manuscript
    if batch_log is not None:
        update["$set"]["batch_log"] = batch_log
    if progress is not None:
        update["$set"]["progress"] = progress
        if state is None:
            update["$set"]["state.progress"] = progress
        update["$push"] = {
            "progress_log": {
                "$each": [progress],
                "$slice": -100,
            }
        }
    db.stories.update_one({"_id": story_id}, update)


def update_story_progress(story_id: str, progress: Dict[str, Any]) -> None:
    """Persist a live progress event for polling UIs and server-side audit logs."""
    db = get_db()
    now = datetime.datetime.utcnow()
    event = {**progress, "updated_at": now}
    db.stories.update_one(
        {"_id": story_id},
        {
            "$set": {
                "progress": event,
                "state.progress": event,
                "updated_at": now,
            },
            "$push": {
                "progress_log": {
                    "$each": [event],
                    "$slice": -100,
                }
            },
        },
    )


def append_batch_log(story_id: str, entry: Dict[str, Any]) -> None:
    """Push a new entry onto the story's ``batch_log`` array."""
    db = get_db()
    db.stories.update_one(
        {"_id": story_id},
        {
            "$push": {"batch_log": entry},
            "$set": {"updated_at": datetime.datetime.utcnow()},
        },
    )


def get_story(story_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single story document by its ID."""
    db = get_db()
    return db.stories.find_one({"_id": story_id})


def list_stories(limit: int = 20, skip: int = 0) -> List[Dict[str, Any]]:
    """List stories (most-recent first), excluding the full manuscript.

    Hidden stories (``hidden == True``) are filtered out.
    """
    db = get_db()
    cursor = (
        db.stories.find({"hidden": {"$ne": True}}, {"final_manuscript": 0})
        .sort("created_at", -1)
        .skip(skip)
        .limit(limit)
    )
    return list(cursor)


def set_story_hidden(story_id: str, hidden: bool) -> Optional[Dict[str, Any]]:
    """Toggle the ``hidden`` flag on a story. Returns the updated document."""
    db = get_db()
    return db.stories.find_one_and_update(
        {"_id": story_id},
        {"$set": {
            "hidden": hidden,
            "updated_at": datetime.datetime.utcnow(),
        }},
        return_document=ReturnDocument.AFTER,
    )


# ---------------------------------------------------------------------------
# Profile CRUD — stores both the inputs AND the generated LLM outputs
# ---------------------------------------------------------------------------

def save_generated_profile(
    profile_id: str,
    name: str,
    *,
    # Input fields (preserved for re-generation)
    bio_context: str = "",
    writing_samples: List[str] | None = None,
    # LLM outputs (the whole point — these get injected into stories)
    author_profile: str,
    emotional_guidelines: str,
    expert_styles: str,
) -> Dict[str, Any]:
    """Store a generated profile with both inputs and LLM outputs."""
    db = get_db()
    doc = {
        "_id": profile_id,
        "name": name,
        "bio_context": bio_context,
        "writing_samples": writing_samples or [],
        "author_profile": author_profile,
        "emotional_guidelines": emotional_guidelines,
        "expert_styles": expert_styles,
        "created_at": datetime.datetime.utcnow(),
        "updated_at": datetime.datetime.utcnow(),
    }
    db.profiles.insert_one(doc)
    return doc


def get_profile(profile_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a profile by ID."""
    db = get_db()
    return db.profiles.find_one({"_id": profile_id})


def list_profiles() -> List[Dict[str, Any]]:
    """List all profiles (without the full writing samples or LLM outputs).

    Hidden profiles (``hidden == True``) are filtered out.
    """
    db = get_db()
    cursor = db.profiles.find(
        {"hidden": {"$ne": True}},
        {
            "writing_samples": 0,
            "author_profile": 0,
            "emotional_guidelines": 0,
            "expert_styles": 0,
        },
    ).sort("created_at", -1)
    return list(cursor)


def set_profile_hidden(profile_id: str, hidden: bool) -> Optional[Dict[str, Any]]:
    """Toggle the ``hidden`` flag on a profile. Returns the updated document."""
    db = get_db()
    return db.profiles.find_one_and_update(
        {"_id": profile_id},
        {"$set": {
            "hidden": hidden,
            "updated_at": datetime.datetime.utcnow(),
        }},
        return_document=ReturnDocument.AFTER,
    )
