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
    extract_continuity_from_imported,
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
    ImportSurvey,
    ProfileV2,
    ProfileV2Input,
    QuickSurvey,
    StoryStatus,
    character_budget_for,
)
from .schemas_v2.survey import ParsedSurveyDraft, PastedNotes
from .serializers import (
    CharacterBiblePatchSerializer,
    DeepSurveySerializer,
    ImportSurveySerializer,
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


# ---------------------------------------------------------------------------
# Import-past-chapters wizard
# ---------------------------------------------------------------------------

_VOICE_SAMPLE_MAX_CHARS = 10_000
_VOICE_SAMPLE_MAX_COUNT = 8


def _select_voice_samples(chapters: List[Dict[str, Any]]) -> List[str]:
    """Pick representative passages from imported chapters for voice training.

    Strategy: take the last 1–2 paragraphs of each chapter (typically the
    most emotionally charged), cap to 8 samples and ~10K total chars.
    """
    samples: List[str] = []
    for ch in chapters:
        text = (ch.get("text") or "").strip()
        if not text:
            continue
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        tail = paragraphs[-2:] if len(paragraphs) >= 2 else paragraphs
        sample = "\n\n".join(tail).strip()
        if sample:
            samples.append(sample)

    # Cap by count first (keep most recent — usually the strongest voice).
    if len(samples) > _VOICE_SAMPLE_MAX_COUNT:
        samples = samples[-_VOICE_SAMPLE_MAX_COUNT:]
    # Then by total characters, dropping oldest until under cap.
    while samples and sum(len(s) for s in samples) > _VOICE_SAMPLE_MAX_CHARS:
        samples.pop(0)
    return samples


def _build_import_envelope(
    *,
    story_id: str,
    quick: QuickSurvey,
    deep: DeepSurvey,
    imported_chapter_count: int,
    continuation_brief: str,
    end_story: bool,
    continuity_ledger: Dict[str, Any],
) -> Dict[str, Any]:
    """Story envelope for the Import wizard. Like ``_build_initial_envelope``
    but with imported chapters as committed canon and the engine pointed at N+1.
    """
    now = datetime.datetime.utcnow()
    progress = {
        "stage": "queued",
        "message": (
            f"{imported_chapter_count} chapter(s) imported as canon. "
            "Bibles built; awaiting generation engine."
        ),
        "percent": 0,
        "completed_chapters": imported_chapter_count,
        "total_chapters": quick.num_chapters,
        "updated_at": now,
    }
    return {
        "_id": story_id,
        "title": quick.title,
        "status": StoryStatus.PENDING.value,
        "quick_survey": quick.model_dump(),
        "deep_survey": deep.model_dump(),
        "arc_plan": None,
        "chapters": [],          # filled by mongo.seed_canon_chapters
        "current_chapter_idx": imported_chapter_count,
        "current_scene_idx": 0,
        "progress": progress,
        "progress_log": [progress],
        "created_at": now,
        "updated_at": now,
        "hidden": False,
        "character_budget": character_budget_for(quick.num_chapters),
        # Engine-facing extras (Phase 8+ reads these):
        "continuation_brief": continuation_brief,
        "end_story_next_batch": bool(end_story),
        "continuity_ledger": continuity_ledger,
        "imported_chapter_count": imported_chapter_count,
    }


@api_view(["POST"])
def create_story_import(request):
    """POST /api/v2/stories/import/

    Import N past chapters as canon + a continuation brief + chapters_to_generate.
    Either references an existing profile or generates one from the chapters.
    Returns the new ``story_id`` so the frontend can route to ``#/stories/<id>/bibles``.
    """
    serializer = ImportSurveySerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    survey: ImportSurvey = serializer.pydantic

    chapter_dicts: List[Dict[str, Any]] = [c.model_dump() for c in survey.chapters]
    imported_count = len(chapter_dicts)
    voice_samples = _select_voice_samples(chapter_dicts)

    # ---- Validate profile selection BEFORE we do any LLM work ----------------
    if survey.profile_mode == "select":
        if not mongo.get_profile(survey.profile_id or ""):
            return Response(
                {"detail": "profile_id does not match any existing profile."},
                status=status.HTTP_400_BAD_REQUEST,
            )

    # ---- Build the combined text once + parser hint --------------------------
    combined_text_parts: List[str] = []
    for i, ch in enumerate(chapter_dicts):
        title = ch.get("title") or f"Chapter {i + 1}"
        combined_text_parts.append(f"--- {title} ---\n{ch.get('text') or ''}")
    combined_text_parts.append("--- AUTHOR'S CONTINUATION BRIEF ---\n" + survey.description)
    combined = "\n\n".join(combined_text_parts)
    parser_hint = (
        f"Above are {imported_count} prior chapter(s) of a story, followed by the "
        "author's brief for what should happen in the upcoming chapters. Extract "
        "characters, world, arc preferences. Treat the chapters as canon."
    )

    # ---- Run ALL agent calls in a single asyncio loop (asgiref's
    # async_to_sync closes its loop after each top-level call, which can
    # invalidate provider HTTP clients held across calls — composing into one
    # coroutine sidesteps that AND lets independent agents run in parallel).
    async def _run_agents() -> Dict[str, Any]:
        import asyncio
        tasks: Dict[str, Any] = {
            "parsed": parse_pasted_notes(
                PastedNotes(raw_text=combined[:200_000], hint=parser_hint)
            ),
            "continuity": extract_continuity_from_imported(chapter_dicts),
        }
        if survey.profile_mode == "generate":
            tasks["profile"] = build_profile(
                ProfileV2Input(
                    name=(survey.new_profile_name or "").strip(),
                    bio_context=(survey.new_profile_bio or "").strip(),
                    writing_samples=voice_samples,
                )
            )
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return dict(zip(tasks.keys(), results))

    try:
        agent_results = async_to_sync(_run_agents)()
    except Exception as exc:
        logger.exception("import: agent batch failed")
        return Response(
            {"ok": False, "error": f"agent batch failed: {type(exc).__name__}: {exc}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    parsed_result = agent_results.get("parsed")
    if isinstance(parsed_result, Exception):
        logger.warning("import: paste parser failed (%s)", parsed_result)
        # Fall through with an empty parse — bibles will be sparse but the
        # canon chapters are still seeded.
        parsed = ParsedSurveyDraft(notes=[f"Parser unavailable: {parsed_result}"])
    else:
        parsed = parsed_result  # type: ignore[assignment]

    # ---- Resolve the profile (after agents have run) -------------------------
    resolved_profile_id: Optional[str] = None
    if survey.profile_mode == "select":
        resolved_profile_id = survey.profile_id
    else:
        prof_result = agent_results.get("profile")
        if isinstance(prof_result, Exception):
            logger.exception("import: profile generation failed", exc_info=prof_result)
            return Response(
                {"ok": False, "error": f"profile generation failed: {type(prof_result).__name__}: {prof_result}"},
                status=status.HTTP_502_BAD_GATEWAY,
            )
        profile_input = ProfileV2Input(
            name=(survey.new_profile_name or "").strip(),
            bio_context=(survey.new_profile_bio or "").strip(),
            writing_samples=voice_samples,
        )
        new_profile_id = str(uuid.uuid4())
        profile = ProfileV2(
            profile_id=new_profile_id,
            name=profile_input.name,
            inputs=profile_input,
            lexical_fingerprint=prof_result["lexical_fingerprint"],
            emotional_defaults=prof_result["emotional_defaults"],
            preferred_phrases=prof_result["preferred_phrases"],
            banned_phrases=prof_result["banned_phrases"],
            few_shot_samples=prof_result["few_shot_samples"] or voice_samples,
            expertise_tags=prof_result["expertise_tags"],
        )
        mongo.insert_profile({"_id": new_profile_id, **profile.model_dump()})
        resolved_profile_id = new_profile_id

    parsed_notes = list(parsed.notes or [])

    # ---- Build Quick + Deep surveys from the parser output -------------------
    parsed_quick = parsed.quick
    title = (survey.title or "").strip()
    if not title:
        title = (parsed_quick.title if parsed_quick and parsed_quick.title else "(Imported story)")

    premise = ""
    if parsed_quick and parsed_quick.premise:
        premise = parsed_quick.premise
    if not premise:
        premise = survey.description[:400]

    total_chapters = imported_count + survey.chapters_to_generate

    # Quick — derive from parsed_quick when available, else conservative defaults.
    quick_kwargs: Dict[str, Any] = {
        "title": title,
        "premise": premise,
        "num_chapters": total_chapters,
        "initial_chapters": survey.chapters_to_generate,
        "profile_id": resolved_profile_id,
    }
    if parsed_quick:
        if parsed_quick.genres:
            quick_kwargs["genres"] = parsed_quick.genres
        if parsed_quick.tone:
            quick_kwargs["tone"] = parsed_quick.tone
        if parsed_quick.pov:
            quick_kwargs["pov"] = parsed_quick.pov
        if parsed_quick.tense:
            quick_kwargs["tense"] = parsed_quick.tense
        if parsed_quick.target_chapter_words:
            quick_kwargs["target_chapter_words"] = parsed_quick.target_chapter_words
    try:
        quick = QuickSurvey(**quick_kwargs)
    except Exception as exc:
        return Response(
            {"detail": "Failed to build derived QuickSurvey.", "error": str(exc)},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Deep — start from parsed bibles, then mirror our voice samples into
    # style_anchors so the writer agent always sees them at draft time.
    deep_payload: Dict[str, Any] = {
        "quick": quick.model_dump(),
        "characters": [c.model_dump() for c in parsed.characters] if parsed.characters else [],
        "world": parsed.world.model_dump() if parsed.world else {},
        "arc_preferences": parsed.arc_preferences.model_dump() if parsed.arc_preferences else {},
        "style_anchors": parsed.style_anchors.model_dump() if parsed.style_anchors else {
            "reference_authors": [],
            "reference_books": [],
            "pasted_sample_passages": [],
        },
    }
    # Merge voice samples into pasted_sample_passages without exploding past
    # any reasonable size. Dedupe by exact match.
    existing_passages = list(deep_payload["style_anchors"].get("pasted_sample_passages") or [])
    for s in voice_samples:
        if s not in existing_passages:
            existing_passages.append(s)
    deep_payload["style_anchors"]["pasted_sample_passages"] = existing_passages

    try:
        deep = DeepSurvey.model_validate(deep_payload)
    except Exception as exc:
        return Response(
            {"detail": "Failed to build derived DeepSurvey.", "error": str(exc)},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # ---- Continuity-extraction (already ran inside _run_agents) --------------
    continuity_result = agent_results.get("continuity")
    if isinstance(continuity_result, Exception) or not isinstance(continuity_result, dict):
        if isinstance(continuity_result, Exception):
            logger.warning("import: continuity extraction failed (%s)", continuity_result)
        continuity = {
            "chapter_summaries": [""] * imported_count,
            "open_threads": [],
            "latest_cliffhanger": None,
            "character_moods": {},
        }
    else:
        continuity = continuity_result
    continuity_ledger = {
        "open_threads": continuity.get("open_threads") or [],
        "latest_cliffhanger": continuity.get("latest_cliffhanger"),
        "unresolved_cliffhangers": [],
    }

    # ---- Build + persist envelope -------------------------------------------
    story_id = str(uuid.uuid4())
    envelope = _build_import_envelope(
        story_id=story_id,
        quick=quick,
        deep=deep,
        imported_chapter_count=imported_count,
        continuation_brief=survey.description,
        end_story=survey.end_story,
        continuity_ledger=continuity_ledger,
    )
    mongo.insert_story_envelope(envelope)

    # Seed canon chapters into scene_drafts + envelope.chapters[].
    try:
        mongo.seed_canon_chapters(
            story_id,
            chapter_dicts,
            summaries=continuity.get("chapter_summaries"),
        )
    except Exception as exc:
        logger.exception("import: seed_canon_chapters failed")
        return Response(
            {"ok": False, "error": f"seeding canon chapters failed: {type(exc).__name__}: {exc}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    # Persist bibles — reuse the same path as Deep wizard.
    try:
        bibles = _persist_bibles(story_id=story_id, quick=quick, deep=deep)
    except Exception as exc:
        logger.exception("import: bible build failed")
        mongo.update_story_envelope(
            story_id,
            {
                "status": StoryStatus.FAILED.value,
                "progress": {"stage": "failed", "error": str(exc)},
            },
        )
        return Response(
            {"story_id": story_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    # Apply per-character mood snapshots from the continuity pass.
    # CharacterMoodSnapshot has extra="forbid" — pass it through the schema
    # so we never write unknown fields the engine would reject at re-validate.
    from .schemas_v2 import CharacterMoodSnapshot, PlutchikVector, SceneEmotionAxes

    character_moods = continuity.get("character_moods") or {}
    if character_moods:
        cast = mongo.list_character_bibles(story_id)
        name_to_id = {(c.get("name") or "").lower(): c.get("character_id") for c in cast}
        for raw_name, mood in character_moods.items():
            char_id = name_to_id.get((raw_name or "").lower())
            if not char_id or not isinstance(mood, dict):
                continue
            try:
                valence = float(mood.get("valence", 0.0))
                arousal = float(mood.get("arousal", 0.5))
            except (TypeError, ValueError):
                continue
            snapshot = CharacterMoodSnapshot(
                chapter_idx=max(0, imported_count - 1),
                scene_idx=0,
                plutchik=PlutchikVector(),  # zeros — we only got valence/arousal
                axes=SceneEmotionAxes(
                    valence=max(-1.0, min(1.0, valence)),
                    arousal=max(0.0, min(1.0, arousal)),
                ),
                last_event_summary=str(mood.get("summary") or "")[:500],
            )
            mongo.col(mongo.COL_CHARACTERS).update_one(
                {"story_id": story_id, "character_id": char_id},
                {"$push": {"mood_state_history": snapshot.model_dump()}},
            )

    return Response(
        {
            "ok": True,
            "story_id": story_id,
            "profile_id": resolved_profile_id,
            "status": StoryStatus.PENDING.value,
            "chapters_imported": imported_count,
            "num_chapters_total": total_chapters,
            "bibles": bibles,
            "parsed_notes": parsed_notes,
            "continuity_summary": {
                "open_threads_count": len(continuity_ledger["open_threads"]),
                "has_cliffhanger": bool(continuity_ledger["latest_cliffhanger"]),
                "moods_captured": len(character_moods),
            },
            "note": (
                "Bibles + canon chapters persisted. Review bibles, then call "
                "/api/v2/stories/<id>/start/ to begin generating chapter "
                f"{imported_count + 1} onward."
            ),
        },
        status=status.HTTP_201_CREATED,
    )


@api_view(["PATCH"])
def patch_imported_chapter_view(request, story_id: str, chapter_idx: int):
    """PATCH /api/v2/stories/<id>/imported-chapter/<idx>/

    Allows the user to edit an imported chapter from the bible-review page
    BEFORE clicking Start. Once the story has been started the chapter
    becomes immutable from this endpoint (use the scene-level edit
    endpoint instead for fine-grained edits during/after generation).
    """
    story = mongo.get_story_envelope(story_id)
    if not story:
        return Response({"detail": "Story not found."}, status=status.HTTP_404_NOT_FOUND)
    if story.get("status") != StoryStatus.PENDING.value:
        return Response(
            {"detail": "Imported chapters can only be edited before the story has been started."},
            status=status.HTTP_409_CONFLICT,
        )

    data = request.data if isinstance(request.data, dict) else {}
    title = data.get("title")
    text = data.get("text")
    if title is None and text is None:
        return Response(
            {"detail": "Provide at least one of 'title' or 'text'."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if text is not None and len(str(text)) < 50:
        return Response(
            {"detail": "Chapter text must be at least 50 characters."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    updated = mongo.update_imported_chapter(
        story_id,
        int(chapter_idx),
        title=str(title) if title is not None else None,
        text=str(text) if text is not None else None,
    )
    if not updated:
        return Response(
            {"detail": "No imported chapter at that index for this story."},
            status=status.HTTP_404_NOT_FOUND,
        )
    return Response({"ok": True, "story_id": story_id, "chapter_idx": int(chapter_idx)})


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
    for idx, ch in enumerate(story.get("chapters") or []):
        chapters.append({
            "chapter_number": ch.get("chapter_number") or (ch.get("chapter_idx", idx) + 1),
            "chapter_idx": ch.get("chapter_idx", idx),
            "title": ch.get("title"),
            "summary": ch.get("summary"),
            "text": ch.get("text"),
            "word_count": ch.get("word_count"),
            "committed_at": ch.get("committed_at"),
            "source": ch.get("source"),
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

    quick = story.get("quick_survey") or {}
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
        # Slim quick_survey fields the SPA needs (profile swap, dialogs).
        "quick_survey": {
            "profile_id": quick.get("profile_id"),
            "num_chapters": quick.get("num_chapters"),
            "initial_chapters": quick.get("initial_chapters"),
        },
        "imported_chapter_count": story.get("imported_chapter_count") or 0,
        "end_story_next_batch": bool(story.get("end_story_next_batch")),
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

    Body (all optional):
        additional_chapters: int   — batch size for this continue
        end_story:           bool  — if true, the engine treats this batch as a
                                     finale (last chapter resolves arcs + sets
                                     status=completed). Persisted on the
                                     envelope so the Phase 8+ engine can read it.
        profile_id:          str   — swap the story's narrative-voice profile
                                     starting from this batch.
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

    body = request.data if isinstance(request.data, dict) else {}
    quick = story.get("quick_survey", {}) or {}
    total = int(quick.get("num_chapters", 0) or 0)
    starting = int(story.get("current_chapter_idx") or 0)

    # Batch size — defaults to 3, grows the total if the user wants more than
    # the original plan allowed (Improvement B from the plan).
    additional = body.get("additional_chapters")
    try:
        batch = int(additional) if additional is not None else None
    except (TypeError, ValueError):
        batch = None
    if batch is None:
        batch = max(1, min(3, total - starting)) if total > starting else 3
    batch = max(1, batch)

    envelope_updates: Dict[str, Any] = {}
    new_total = max(total, starting + batch)
    if new_total != total:
        envelope_updates["quick_survey.num_chapters"] = new_total

    # Optional profile swap.
    new_profile_id = body.get("profile_id")
    if new_profile_id and new_profile_id != quick.get("profile_id"):
        if not mongo.get_profile(new_profile_id):
            return Response(
                {"detail": "profile_id does not match any existing profile."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        envelope_updates["quick_survey.profile_id"] = new_profile_id

    # Optional end-story flag.
    end_story = body.get("end_story")
    if end_story is not None:
        envelope_updates["end_story_next_batch"] = bool(end_story)

    if envelope_updates:
        mongo.update_story_envelope(story_id, envelope_updates)

    launch_story_run(story_id, batch_size=batch)
    return Response(
        {
            "ok": True,
            "story_id": story_id,
            "status": "queued",
            "batch_size": batch,
            "starting_chapter_idx": starting,
            "num_chapters_total": new_total,
            "end_story_next_batch": bool(end_story) if end_story is not None else bool(story.get("end_story_next_batch")),
            "profile_id": envelope_updates.get("quick_survey.profile_id", quick.get("profile_id")),
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
