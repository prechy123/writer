"""PyMongo connection management and CRUD operations for story documents."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from django.conf import settings
from pymongo import MongoClient

_client: Optional[MongoClient] = None


def get_db():
    """Return the MongoDB database handle (lazy-initialised singleton)."""
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGODB_URI)
    return _client[settings.MONGODB_DB_NAME]


# ---------------------------------------------------------------------------
# Story CRUD
# ---------------------------------------------------------------------------

def create_story_record(
    story_id: str,
    title: str,
    description: str,
    num_chapters: int,
) -> Dict[str, Any]:
    """Insert an initial story document with status='pending'."""
    db = get_db()
    doc = {
        "_id": story_id,
        "title": title,
        "description": description,
        "num_chapters": num_chapters,
        "status": "pending",
        "created_at": datetime.datetime.utcnow(),
        "updated_at": datetime.datetime.utcnow(),
        "state": {},
        "final_manuscript": None,
    }
    db.stories.insert_one(doc)
    return doc


def update_story_status(
    story_id: str,
    status: str,
    *,
    state: Optional[Dict[str, Any]] = None,
    manuscript: Optional[Dict[str, Any]] = None,
) -> None:
    """Update the status (and optionally the full state / manuscript)."""
    db = get_db()
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
    db.stories.update_one({"_id": story_id}, update)


def get_story(story_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single story document by its ID."""
    db = get_db()
    return db.stories.find_one({"_id": story_id})


def list_stories(limit: int = 20, skip: int = 0) -> List[Dict[str, Any]]:
    """List stories (most-recent first), excluding the full manuscript."""
    db = get_db()
    cursor = (
        db.stories.find({}, {"final_manuscript": 0})
        .sort("created_at", -1)
        .skip(skip)
        .limit(limit)
    )
    return list(cursor)


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
    """List all profiles (without the full writing samples or LLM outputs)."""
    db = get_db()
    cursor = db.profiles.find(
        {},
        {
            "writing_samples": 0,
            "author_profile": 0,
            "emotional_guidelines": 0,
            "expert_styles": 0,
        },
    ).sort("created_at", -1)
    return list(cursor)
