"""PyMongo accessors for the v2 collections.

Mirrors the pattern in ``stories/mongodb.py`` (singleton client, lazy
connect, ``certifi`` TLS for Atlas/SRV URIs) but every collection name
is suffixed ``_v2`` so v1 data is untouched.

Collections:
    profiles_v2
    stories_v2
    character_bibles_v2
    world_bibles_v2
    beat_sheets_v2
    scene_drafts_v2
    run_events_v2          (capped, for SSE replay)
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional

import certifi
from django.conf import settings
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import CollectionInvalid, OperationFailure

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None

# Names are exported as constants so other modules don't string-literal them.
COL_PROFILES = "profiles_v2"
COL_STORIES = "stories_v2"
COL_CHARACTERS = "character_bibles_v2"
COL_WORLDS = "world_bibles_v2"
COL_BEATS = "beat_sheets_v2"
COL_SCENES = "scene_drafts_v2"
COL_EVENTS = "run_events_v2"

_RUN_EVENTS_CAP_BYTES = 10 * 1024 * 1024  # 10 MB capped collection
_RUN_EVENTS_TTL_DAYS = 7


def get_db():
    """Return the MongoDB database handle (lazy-init singleton)."""
    global _client
    if _client is None:
        uri = settings.MONGODB_URI
        kwargs: Dict[str, Any] = {}
        if uri.startswith("mongodb+srv://") or "tls=true" in uri.lower():
            kwargs["tlsCAFile"] = certifi.where()
        _client = MongoClient(uri, **kwargs)
    return _client[settings.MONGODB_DB_NAME]


def col(name: str) -> Collection:
    return get_db()[name]


# ---------------------------------------------------------------------------
# Atlas detection — drives memory tier choice (vector search vs. in-memory)
# ---------------------------------------------------------------------------

_atlas_detected: Optional[bool] = None


def is_atlas() -> bool:
    """Return ``True`` if the cluster looks like MongoDB Atlas.

    Atlas detection enables ``$vectorSearch``-backed episodic memory. The
    environment override ``MONGODB_ATLAS_VECTOR=1`` forces enable, useful
    for testing against a self-hosted cluster with vector indexes.
    """
    global _atlas_detected
    if _atlas_detected is not None:
        return _atlas_detected
    import os

    if os.getenv("MONGODB_ATLAS_VECTOR", "").lower() in ("1", "true", "yes"):
        _atlas_detected = True
        return _atlas_detected
    uri = getattr(settings, "MONGODB_URI", "") or ""
    if "mongodb.net" in uri or uri.startswith("mongodb+srv://"):
        _atlas_detected = True
        return _atlas_detected
    _atlas_detected = False
    return _atlas_detected


# ---------------------------------------------------------------------------
# Index + capped-collection bootstrap (idempotent — safe to call repeatedly)
# ---------------------------------------------------------------------------

def ensure_indexes() -> None:
    """Create indexes + capped collection.  Safe to call multiple times."""
    db = get_db()

    # stories_v2
    col(COL_STORIES).create_index([("status", ASCENDING), ("updated_at", DESCENDING)])
    col(COL_STORIES).create_index([("hidden", ASCENDING), ("updated_at", DESCENDING)])

    # character_bibles_v2 — one doc per (story_id, character_id)
    col(COL_CHARACTERS).create_index(
        [("story_id", ASCENDING), ("character_id", ASCENDING)], unique=True
    )
    col(COL_CHARACTERS).create_index([("story_id", ASCENDING), ("tier", ASCENDING)])

    # world_bibles_v2 — one doc per story
    col(COL_WORLDS).create_index([("story_id", ASCENDING)], unique=True)

    # beat_sheets_v2 — one doc per (story_id, chapter_idx)
    col(COL_BEATS).create_index(
        [("story_id", ASCENDING), ("chapter_idx", ASCENDING)], unique=True
    )

    # scene_drafts_v2 — one doc per (story_id, chapter_idx, scene_idx)
    col(COL_SCENES).create_index(
        [
            ("story_id", ASCENDING),
            ("chapter_idx", ASCENDING),
            ("scene_idx", ASCENDING),
        ],
        unique=True,
    )

    # run_events_v2 — capped, monotonic seq per story
    if COL_EVENTS not in db.list_collection_names():
        try:
            db.create_collection(
                COL_EVENTS,
                capped=True,
                size=_RUN_EVENTS_CAP_BYTES,
                max=200_000,
            )
        except CollectionInvalid:
            pass
    col(COL_EVENTS).create_index([("story_id", ASCENDING), ("seq", ASCENDING)])
    # TTL is best-effort — capped collections drop oldest automatically too.
    try:
        col(COL_EVENTS).create_index(
            "created_at",
            expireAfterSeconds=_RUN_EVENTS_TTL_DAYS * 24 * 3600,
        )
    except OperationFailure:
        # Can't add TTL on a capped collection on some Mongo versions; OK.
        pass

    logger.info("stories_v2: ensured indexes on all collections")


# ---------------------------------------------------------------------------
# Story envelope CRUD
# ---------------------------------------------------------------------------

def insert_story_envelope(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Insert a new story document. Caller supplies ``_id``."""
    col(COL_STORIES).insert_one(doc)
    return doc


def get_story_envelope(story_id: str) -> Optional[Dict[str, Any]]:
    return col(COL_STORIES).find_one({"_id": story_id})


def list_story_envelopes(*, limit: int = 20, skip: int = 0) -> List[Dict[str, Any]]:
    cursor = (
        col(COL_STORIES)
        .find({"hidden": {"$ne": True}})
        .sort("created_at", DESCENDING)
        .skip(skip)
        .limit(limit)
    )
    return list(cursor)


def update_story_envelope(story_id: str, fields: Dict[str, Any]) -> None:
    fields = {**fields, "updated_at": datetime.datetime.utcnow()}
    col(COL_STORIES).update_one({"_id": story_id}, {"$set": fields})


def set_story_hidden(story_id: str, hidden: bool) -> Optional[Dict[str, Any]]:
    return col(COL_STORIES).find_one_and_update(
        {"_id": story_id},
        {"$set": {"hidden": hidden, "updated_at": datetime.datetime.utcnow()}},
        return_document=True,
    )


# ---------------------------------------------------------------------------
# Run events — used by SSE streaming + reconnect replay
# ---------------------------------------------------------------------------

def append_run_event(
    story_id: str,
    seq: int,
    event_type: str,
    payload: Dict[str, Any],
) -> None:
    col(COL_EVENTS).insert_one(
        {
            "story_id": story_id,
            "seq": seq,
            "event_type": event_type,
            "payload": payload,
            "created_at": datetime.datetime.utcnow(),
        }
    )


def fetch_run_events_after(
    story_id: str,
    *,
    after_seq: int,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    cursor = (
        col(COL_EVENTS)
        .find({"story_id": story_id, "seq": {"$gt": after_seq}})
        .sort("seq", ASCENDING)
        .limit(limit)
    )
    return list(cursor)


def latest_event_seq(story_id: str) -> int:
    """Return the highest seq number seen for this story (0 if none)."""
    doc = col(COL_EVENTS).find_one(
        {"story_id": story_id}, sort=[("seq", DESCENDING)], projection={"seq": 1}
    )
    return int((doc or {}).get("seq") or 0)


# ---------------------------------------------------------------------------
# Profile v2 CRUD
# ---------------------------------------------------------------------------

def insert_profile(doc: Dict[str, Any]) -> Dict[str, Any]:
    col(COL_PROFILES).insert_one(doc)
    return doc


def get_profile(profile_id: str) -> Optional[Dict[str, Any]]:
    return col(COL_PROFILES).find_one({"_id": profile_id})


def list_profiles(*, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
    cursor = (
        col(COL_PROFILES)
        .find({"hidden": {"$ne": True}}, {"few_shot_samples": 0})
        .sort("created_at", DESCENDING)
        .skip(skip)
        .limit(limit)
    )
    return list(cursor)


def update_profile(profile_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fields = {**fields, "updated_at": datetime.datetime.utcnow()}
    return col(COL_PROFILES).find_one_and_update(
        {"_id": profile_id},
        {"$set": fields},
        return_document=True,
    )


def set_profile_hidden(profile_id: str, hidden: bool) -> Optional[Dict[str, Any]]:
    return col(COL_PROFILES).find_one_and_update(
        {"_id": profile_id},
        {"$set": {"hidden": hidden, "updated_at": datetime.datetime.utcnow()}},
        return_document=True,
    )


# ---------------------------------------------------------------------------
# Character bible v2 CRUD
# ---------------------------------------------------------------------------

def insert_character_bible(doc: Dict[str, Any]) -> Dict[str, Any]:
    col(COL_CHARACTERS).insert_one(doc)
    return doc


def insert_character_bibles_bulk(docs: List[Dict[str, Any]]) -> None:
    if not docs:
        return
    col(COL_CHARACTERS).insert_many(docs, ordered=False)


def get_character_bible(story_id: str, character_id: str) -> Optional[Dict[str, Any]]:
    return col(COL_CHARACTERS).find_one(
        {"story_id": story_id, "character_id": character_id}
    )


def list_character_bibles(story_id: str, *, tier: Optional[str] = None) -> List[Dict[str, Any]]:
    query: Dict[str, Any] = {"story_id": story_id}
    if tier:
        query["tier"] = tier
    cursor = col(COL_CHARACTERS).find(query).sort("tier", ASCENDING)
    return list(cursor)


def update_character_bible(
    story_id: str,
    character_id: str,
    fields: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    fields = {**fields, "updated_at": datetime.datetime.utcnow()}
    return col(COL_CHARACTERS).find_one_and_update(
        {"story_id": story_id, "character_id": character_id},
        {"$set": fields},
        return_document=True,
    )


# ---------------------------------------------------------------------------
# World bible v2 CRUD
# ---------------------------------------------------------------------------

def upsert_world_bible(story_id: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    doc = {**doc, "story_id": story_id, "updated_at": datetime.datetime.utcnow()}
    col(COL_WORLDS).update_one({"story_id": story_id}, {"$set": doc}, upsert=True)
    persisted = col(COL_WORLDS).find_one({"story_id": story_id})
    return persisted or doc


def get_world_bible(story_id: str) -> Optional[Dict[str, Any]]:
    return col(COL_WORLDS).find_one({"story_id": story_id})


def update_world_bible(story_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fields = {**fields, "updated_at": datetime.datetime.utcnow()}
    return col(COL_WORLDS).find_one_and_update(
        {"story_id": story_id},
        {"$set": fields},
        return_document=True,
    )
