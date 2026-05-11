"""Semantic memory — always-injected bibles + active world rules.

The Scene Writer always sees:
  - Full bible for every character listed in present_characters.
  - World rules tagged 'global' OR whose scope appears in the scene location.
  - Active plot threads (open, not resolved).
  - Unresolved cliffhangers.

This is the "background knowledge" tier — no retrieval, no embeddings,
just direct fetches.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .. import mongo


def fetch_present_characters(
    *,
    story_id: str,
    character_ids: List[str],
) -> List[Dict[str, Any]]:
    """Return full bibles for the named characters. Drops side-tier mood
    history because side characters don't have one."""
    if not character_ids:
        return []
    docs = list(
        mongo.col(mongo.COL_CHARACTERS).find(
            {"story_id": story_id, "character_id": {"$in": character_ids}},
            projection={"_id": 0},
        )
    )
    return docs


def fetch_world_bible(*, story_id: str) -> Optional[Dict[str, Any]]:
    return mongo.col(mongo.COL_WORLDS).find_one({"story_id": story_id}, {"_id": 0})


def relevant_world_rules(
    world_bible: Optional[Dict[str, Any]],
    *,
    scene_location: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter world rules to global + scope-matching."""
    if not world_bible:
        return []
    rules = world_bible.get("rules") or []
    if not scene_location:
        return list(rules)

    loc = scene_location.lower()
    out: List[Dict[str, Any]] = []
    for rule in rules:
        scope = (rule.get("scope") or "").lower()
        if not scope or scope == "global":
            out.append(rule)
        elif scope in loc or loc in scope:
            out.append(rule)
    return out


def fetch_active_threads(*, story_id: str) -> Dict[str, List[Any]]:
    """Look up open plot threads / cliffhangers from the story envelope's
    continuity_ledger if it exists. Returns empty lists if no ledger yet
    (story still in early generation)."""
    story = mongo.col(mongo.COL_STORIES).find_one(
        {"_id": story_id},
        projection={"continuity_ledger": 1},
    )
    ledger = (story or {}).get("continuity_ledger") or {}
    return {
        "open_threads": ledger.get("open_threads") or [],
        "unresolved_cliffhangers": ledger.get("unresolved_cliffhangers") or [],
        "plot_seeds": ledger.get("plot_seeds") or [],
        "subplots": [s for s in (ledger.get("subplots") or []) if s.get("status") != "resolved"],
    }
