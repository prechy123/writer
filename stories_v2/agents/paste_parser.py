"""Paste-anything parser — extracts a ParsedSurveyDraft from free text.

LLMs drift from JSON schemas even when prompted strictly. Rather than
fail the whole parse on a single wrong-typed field, we coerce common
drifts back to the schema before validating:

  - age as int → str
  - world.rules as ["string", ...] → [{"scope": "global", "rule": "...", "consequence_if_broken": ""}, ...]
  - world.factions as ["string", ...] → [{"name": "string"}, ...]
  - unknown character/world top-level keys → folded into background/notes or dropped
  - empty strings on required fields like quick.title → field removed
  - magic_or_system as string → {"description": "<string>"}

Anything still too broken to coerce falls back to a ``ParsedSurveyDraft``
with that subfield omitted, so the rest of the parse still lands. The
last-resort empty-with-note response is reserved for total LLM failure.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..prompts_v2.paste_parser import SYSTEM, build_user_prompt
from ..providers import Router, get_router
from ..schemas_v2 import ParsedSurveyDraft, PastedNotes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coercion: normalise LLM drift into the strict pydantic shape
# ---------------------------------------------------------------------------

# Fields the strict DeepCharacter schema accepts. Anything else gets either
# folded into ``background`` (free text we don't want to lose) or dropped.
_DEEP_CHARACTER_KEYS = {
    "name", "tier", "role", "pronouns", "age",
    "background", "motivations", "fears", "secrets", "arc",
    "education_access", "resources_or_limitations", "knowledge_sources",
    "relationships",
    "speech_traits", "preferred_phrases", "banned_phrases", "sample_lines",
    "webnovel_role_hook", "progression_function",
}
# Character extras the LLM commonly invents — we fold the value into background.
_CHARACTER_FOLD_INTO_BACKGROUND = {
    "backstory", "history", "story", "biography", "description",
    "traits", "personality", "conflicts", "tensions", "weaknesses",
    "strengths", "skills", "abilities", "powers",
}

_DEEP_WORLD_KEYS = {
    "setting", "time_period", "technology_level", "social_structure",
    "geography", "languages", "factions", "magic_or_system", "rules",
    "banned_anachronisms", "must_have_vibes", "notes",
}
_WORLD_FOLD_INTO_NOTES = {
    "culture", "religion", "economy", "history", "politics", "climate",
    "atmosphere", "tone",
}

_QUICK_KEYS = {
    "title", "premise", "num_chapters", "initial_chapters",
    "genres", "tone", "characters",
    "pov", "tense", "target_chapter_words", "profile_id",
}

_MAGIC_KINDS = {"none", "cultivation", "litrpg", "classical_magic", "psionics", "tech", "other"}


def _as_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s or None
    if isinstance(v, (int, float, bool)):
        return str(v)
    return None


def _as_list_of_str(v: Any) -> List[str]:
    if not isinstance(v, list):
        return []
    out: List[str] = []
    for item in v:
        s = _as_str(item)
        if s:
            out.append(s)
    return out


def _coerce_character(raw: Any) -> Optional[Dict[str, Any]]:
    """Coerce one character dict to DeepCharacter-compatible shape."""
    if not isinstance(raw, dict):
        return None
    out: Dict[str, Any] = {}
    fold_bits: List[str] = []

    for key, val in raw.items():
        k = str(key).lower().strip()
        if k in _DEEP_CHARACTER_KEYS:
            out[k] = val
        elif k in _CHARACTER_FOLD_INTO_BACKGROUND:
            s = _as_str(val) if not isinstance(val, list) else ", ".join(_as_list_of_str(val))
            if s:
                fold_bits.append(f"{k.capitalize()}: {s}")
        # else: silently drop

    name = _as_str(out.get("name"))
    if not name:
        return None  # No name → useless
    out["name"] = name

    # tier
    tier = _as_str(out.get("tier")) or "recurring"
    tier = tier.lower()
    if tier not in {"main", "recurring", "side"}:
        tier = "recurring"
    out["tier"] = tier

    # age is a STRING in the schema
    if "age" in out:
        s = _as_str(out["age"])
        if s is None:
            out.pop("age", None)
        else:
            out["age"] = s

    # Coerce free-string fields
    for key in ("role", "pronouns", "background", "arc", "education_access",
                "resources_or_limitations", "speech_traits",
                "webnovel_role_hook", "progression_function"):
        if key in out:
            s = _as_str(out[key])
            if s is None:
                out.pop(key, None)
            else:
                out[key] = s

    # List-of-string fields
    for key in ("motivations", "fears", "secrets", "knowledge_sources",
                "preferred_phrases", "banned_phrases", "sample_lines"):
        if key in out:
            out[key] = _as_list_of_str(out[key])

    # Relationships — keep only dicts that have other_character_id-ish shape.
    if "relationships" in out:
        rels = out["relationships"]
        if not isinstance(rels, list):
            out.pop("relationships", None)
        else:
            clean: List[Dict[str, Any]] = []
            for r in rels:
                if not isinstance(r, dict):
                    continue
                other_id = _as_str(r.get("other_character_id")) or _as_str(r.get("other_name"))
                if not other_id:
                    continue
                clean.append({
                    "other_character_id": other_id,
                    "other_name": _as_str(r.get("other_name")),
                    "nature": (_as_str(r.get("nature")) or "acquaintance").lower(),
                    "tension": int(r.get("tension") or 5) if isinstance(r.get("tension"), (int, float, str)) else 5,
                    "history": _as_str(r.get("history")) or "",
                })
            out["relationships"] = clean

    # Fold extras into background
    if fold_bits:
        existing_bg = _as_str(out.get("background")) or ""
        merged = ". ".join(p for p in [existing_bg.rstrip(". ")] + fold_bits if p)
        out["background"] = merged[:4000]

    return out


def _coerce_world_rules(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append({"scope": "global", "rule": text, "consequence_if_broken": ""})
        elif isinstance(item, dict):
            rule_text = _as_str(item.get("rule")) or _as_str(item.get("text"))
            if not rule_text:
                continue
            out.append({
                "scope": _as_str(item.get("scope")) or "global",
                "rule": rule_text,
                "consequence_if_broken": _as_str(item.get("consequence_if_broken")) or "",
            })
    return out


def _coerce_world_factions(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            name = item.strip()
            if name:
                out.append({"name": name})
        elif isinstance(item, dict):
            name = _as_str(item.get("name"))
            if not name:
                continue
            out.append({
                "name": name,
                "role": _as_str(item.get("role")) or "",
                "goals": _as_list_of_str(item.get("goals")),
                "methods": _as_list_of_str(item.get("methods")),
                "notable_members": _as_list_of_str(item.get("notable_members")),
                "relationship_to_protagonist": _as_str(item.get("relationship_to_protagonist")) or "",
            })
    return out


def _coerce_magic(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        lower = s.lower()
        if lower in _MAGIC_KINDS:
            return {"kind": lower, "description": ""}
        return {"kind": "other", "description": s[:4000]}
    if not isinstance(raw, dict):
        return None
    kind = (_as_str(raw.get("kind")) or "none").lower()
    if kind not in _MAGIC_KINDS:
        kind = "other"
    return {
        "kind": kind,
        "description": _as_str(raw.get("description")) or "",
        "progression_path": _as_list_of_str(raw.get("progression_path")),
        "cost_or_drawback": _as_str(raw.get("cost_or_drawback")) or "",
        "hard_limits": _as_list_of_str(raw.get("hard_limits")),
    }


def _coerce_world(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    out: Dict[str, Any] = {}
    extra_notes: List[str] = []

    for key, val in raw.items():
        k = str(key).lower().strip()
        if k in _DEEP_WORLD_KEYS:
            out[k] = val
        elif k in _WORLD_FOLD_INTO_NOTES:
            s = _as_str(val) if not isinstance(val, list) else ", ".join(_as_list_of_str(val))
            if s:
                extra_notes.append(f"{k.capitalize()}: {s}")
        # else: silently drop

    # Free-text fields
    for key in ("setting", "time_period", "technology_level", "social_structure", "geography"):
        if key in out:
            s = _as_str(out[key])
            if s is None:
                out.pop(key, None)
            else:
                out[key] = s

    if "languages" in out:
        out["languages"] = _as_list_of_str(out["languages"])
    if "banned_anachronisms" in out:
        out["banned_anachronisms"] = _as_list_of_str(out["banned_anachronisms"])
    if "must_have_vibes" in out:
        out["must_have_vibes"] = _as_list_of_str(out["must_have_vibes"])

    if "factions" in out:
        out["factions"] = _coerce_world_factions(out["factions"])
    if "rules" in out:
        out["rules"] = _coerce_world_rules(out["rules"])
    if "magic_or_system" in out:
        coerced = _coerce_magic(out["magic_or_system"])
        if coerced is None:
            out.pop("magic_or_system", None)
        else:
            out["magic_or_system"] = coerced

    if "notes" in out:
        s = _as_str(out["notes"])
        if s is None:
            out.pop("notes", None)
        else:
            out["notes"] = s

    if extra_notes:
        existing_notes = _as_str(out.get("notes")) or ""
        merged = ". ".join(p for p in [existing_notes.rstrip(". ")] + extra_notes if p)
        out["notes"] = merged

    return out or None


def _coerce_quick(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    out: Dict[str, Any] = {}
    for key, val in raw.items():
        k = str(key).lower().strip()
        if k in _QUICK_KEYS:
            out[k] = val
    # title: drop if empty string
    if "title" in out:
        s = _as_str(out["title"])
        if s is None:
            out.pop("title", None)
        else:
            out["title"] = s
    if "premise" in out:
        s = _as_str(out["premise"])
        if s is None or len(s) < 10:
            out.pop("premise", None)
        else:
            out["premise"] = s
    # target_chapter_words must be 600..8000 — drop if outside.
    if "target_chapter_words" in out:
        try:
            n = int(out["target_chapter_words"])
        except (TypeError, ValueError):
            n = 0
        if 600 <= n <= 8000:
            out["target_chapter_words"] = n
        else:
            out.pop("target_chapter_words", None)
    # num_chapters: clamp to schema range or drop.
    if "num_chapters" in out:
        try:
            n = int(out["num_chapters"])
        except (TypeError, ValueError):
            n = 0
        if 1 <= n <= 200:
            out["num_chapters"] = n
        else:
            out.pop("num_chapters", None)
    if "genres" in out:
        out["genres"] = _as_list_of_str(out["genres"])
    if "tone" in out:
        out["tone"] = _as_list_of_str(out["tone"])

    # If we have *any* meaningful field, fill required-but-missing fields
    # with placeholders so the strict QuickSurvey schema accepts the dict.
    # The Import wizard's view overrides title + num_chapters with the user's
    # actual inputs; the parsed quick is only consulted for optional hints
    # (genres, tone, pov, tense, target_chapter_words).
    if out:
        out.setdefault("title", "(parsed draft)")
        out.setdefault("num_chapters", 10)
    return out or None


def _coerce_arc_preferences(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    out: Dict[str, Any] = {}
    for key in ("must_include_tropes", "must_avoid_tropes",
                "progression_milestones", "plot_seeds_to_plant"):
        if key in raw:
            out[key] = _as_list_of_str(raw[key])
    journey = _as_str(raw.get("reader_emotional_journey"))
    if journey:
        out["reader_emotional_journey"] = journey
    for key, allowed in (
        ("cliffhanger_intensity", {"low", "medium", "high"}),
        ("pacing_speed", {"slow_burn", "balanced", "breakneck"}),
        ("romance_temperature", {"none", "subtext", "warm", "spicy"}),
        ("action_density", {"light", "balanced", "heavy"}),
    ):
        v = (_as_str(raw.get(key)) or "").lower()
        if v in allowed:
            out[key] = v
    return out or None


def _coerce_style_anchors(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    return {
        "reference_authors": _as_list_of_str(raw.get("reference_authors")),
        "reference_books": _as_list_of_str(raw.get("reference_books")),
        "pasted_sample_passages": _as_list_of_str(raw.get("pasted_sample_passages")),
    }


def _coerce_parsed_draft(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise the raw LLM output into the strict ParsedSurveyDraft shape.

    Any subfield that comes back too broken to recover is dropped. Anything
    we successfully coerced lands. Counts what we had to fix.
    """
    if not isinstance(raw, dict):
        return {"notes": [f"Parser produced a non-object payload ({type(raw).__name__})."]}

    out: Dict[str, Any] = {}
    coercions: List[str] = []

    quick = _coerce_quick(raw.get("quick"))
    if quick:
        out["quick"] = quick

    chars_raw = raw.get("characters") or []
    if isinstance(chars_raw, list):
        chars: List[Dict[str, Any]] = []
        for c in chars_raw:
            coerced = _coerce_character(c)
            if coerced:
                chars.append(coerced)
        if chars:
            out["characters"] = chars
            if len(chars) != len(chars_raw):
                coercions.append(f"dropped {len(chars_raw) - len(chars)} unparseable character entries")

    world = _coerce_world(raw.get("world"))
    if world:
        out["world"] = world

    arc = _coerce_arc_preferences(raw.get("arc_preferences"))
    if arc:
        out["arc_preferences"] = arc

    style = _coerce_style_anchors(raw.get("style_anchors"))
    if style and any(style.values()):
        out["style_anchors"] = style

    notes = _as_list_of_str(raw.get("notes"))
    if coercions:
        notes = notes + [f"Parser coercion: {note}" for note in coercions]
    if notes:
        out["notes"] = notes

    return out


# ---------------------------------------------------------------------------
# Public agent
# ---------------------------------------------------------------------------

async def parse_pasted_notes(
    notes: PastedNotes,
    *,
    router: Optional[Router] = None,
) -> ParsedSurveyDraft:
    router = router or get_router()
    prompt = build_user_prompt(raw_text=notes.raw_text, hint=notes.hint)

    raw: Dict[str, Any]
    try:
        raw = await router.chat_json(
            role="profiler",  # reuse a roomy long-context role for parsing
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.3,
        )
    except Exception as exc:
        logger.warning("paste_parser: LLM call failed (%s)", exc)
        return ParsedSurveyDraft(notes=[f"Parser unavailable: {exc}"])

    coerced = _coerce_parsed_draft(raw)
    try:
        return ParsedSurveyDraft.model_validate(coerced)
    except Exception as exc:
        # Coercion missed something — degrade gracefully, keeping any notes
        # we accumulated so far. The user's canon chapters still ship.
        logger.warning("paste_parser: schema validation failed after coercion (%s)", exc)
        return ParsedSurveyDraft(notes=[
            f"Parser produced output we couldn't coerce: {exc}",
            *(coerced.get("notes") or []),
        ])
