"""LangGraph node functions for all eight story-writing agents.

Each function has the signature ``async def node(state: StoryState) -> dict``
and returns a *partial* state update that LangGraph merges into the global
state.

Phase 1 nodes (profiler, empath, masterclass) check whether their output
field is already populated.  If so they return immediately — this is the
fast-path when a stored profile is used.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from django.conf import settings as django_settings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_together import ChatTogether

from .prompts import (
    continuity_extractor_prompt,
    empath_prompt,
    launch_chapter_planner_prompt,
    masterclass_prompt,
    perfectionist_prompt,
    profiler_prompt,
    publisher_prompt,
    reviewer_prompt,
    storyteller_prompt,
    writer_prompt,
)
from .schemas import (
    ContinuityLedger,
    LaunchChapterPlan,
    ReviewVerdict,
    StoryPlanSchema,
)
from .mongodb import update_story_progress
from .retrieval import (
    build_retrieval_query,
    embed_text,
    select_relevant_summaries,
)
from .state import StoryState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _model_for_role(role: str = "default") -> str:
    """Resolve the Together serverless model for a pipeline role."""
    model_map = getattr(django_settings, "TOGETHER_MODELS", {}) or {}
    default_model = (
        model_map.get("default")
        or getattr(django_settings, "TOGETHER_DEFAULT_MODEL", "")
        or getattr(django_settings, "TOGETHER_MODEL", "")
    )
    return model_map.get(role) or default_model


def _max_tokens_for_role(role: str = "default") -> int | None:
    """Resolve the Together output token budget for a pipeline role."""
    token_map = getattr(django_settings, "TOGETHER_MAX_TOKENS", {}) or {}
    if role in token_map:
        configured = token_map[role]
    elif "default" in token_map:
        configured = token_map["default"]
    else:
        configured = getattr(django_settings, "TOGETHER_DEFAULT_MAX_TOKENS", None)
    if configured is None:
        return None
    try:
        value = int(configured)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid Together max token value for role '%s': %r",
            role,
            configured,
        )
        return None
    return value if value > 0 else None


def _get_llm(
    temperature: float = 0.7,
    max_retries: int = 3,
    role: str = "default",
) -> ChatTogether:
    model = _model_for_role(role)
    max_tokens = _max_tokens_for_role(role)
    logger.debug(
        "Together model selected for role '%s': %s (max_tokens=%s)",
        role,
        model,
        max_tokens,
    )
    return ChatTogether(
        api_key=django_settings.TOGETHER_API_KEY,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        timeout=120,
    )


def _chapter_percent(
    completed_chapters: int,
    total_chapters: int,
    phase_fraction: float = 0.0,
) -> int:
    """Map chapter work onto a simple 20-95% progress range."""
    if total_chapters <= 0:
        return 0
    chapter_span = 75 / total_chapters
    percent = 20 + (completed_chapters * chapter_span) + (phase_fraction * chapter_span)
    return max(0, min(99, int(percent)))


def _mark_progress(
    state: StoryState,
    stage: str,
    message: str,
    *,
    chapter_number: int | None = None,
    percent: int | None = None,
    phase_fraction: float = 0.0,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Log and persist the current graph step without interrupting generation."""
    total = len(state.get("chapters_to_write") or []) or int(state.get("num_chapters", 0) or 0)
    current_idx = int(state.get("current_chapter_index", 0) or 0)
    completed = len(state.get("final_chapters") or []) or current_idx
    target = int(state.get("target_chapter_index") or total or 0)
    if chapter_number is None and total and current_idx < total:
        chapter_number = current_idx + 1
    if percent is None:
        percent = _chapter_percent(completed, total, phase_fraction)

    progress = {
        "stage": stage,
        "message": message,
        "current_chapter": chapter_number,
        "current_chapter_index": current_idx,
        "completed_chapters": completed,
        "target_chapter_index": target,
        "total_chapters": total,
        "percent": percent,
    }
    if extra:
        progress.update(extra)

    story_id = state.get("story_id")
    if story_id:
        logger.info(
            "Story %s — progress [%s] %s (chapter=%s completed=%d/%d percent=%s)",
            story_id,
            stage,
            message,
            chapter_number or "-",
            completed,
            total,
            percent,
        )
        try:
            update_story_progress(story_id, progress)
        except Exception:
            logger.exception("Story %s — failed to persist progress update", story_id)
    else:
        logger.info("Progress [%s] %s", stage, message)
    return progress


# ---------------------------------------------------------------------------
# Phase 1 — parallel context builders (skip if already populated)
# ---------------------------------------------------------------------------

async def profiler_node(state: StoryState) -> Dict[str, Any]:
    """Agent 1: generates the author voice profile.

    Skips the LLM call if ``author_profile`` is already set (i.e. loaded
    from a stored profile).
    """
    if state.get("author_profile"):
        logger.info("Profiler skipped — using stored profile")
        _mark_progress(
            state,
            "profile",
            "Author voice profile already available.",
            percent=5,
        )
        return {}

    _mark_progress(
        state,
        "profile",
        "Building author voice profile.",
        percent=5,
    )
    llm = _get_llm(temperature=0.7, role="profile")
    system = profiler_prompt(state["book_title"], state["book_description"])
    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(
            content=f"Create a detailed author voice profile for: {state['book_title']}"
        ),
    ])
    logger.info("Profiler complete (%d chars)", len(response.content))
    return {"author_profile": response.content}


async def empath_node(state: StoryState) -> Dict[str, Any]:
    """Agent 2: generates emotional guidelines.

    Skips the LLM call if ``emotional_guidelines`` is already set.
    """
    if state.get("emotional_guidelines"):
        logger.info("Empath skipped — using stored profile")
        _mark_progress(
            state,
            "empath",
            "Emotional guidelines already available.",
            percent=8,
        )
        return {}

    _mark_progress(
        state,
        "empath",
        "Building emotional guidelines.",
        percent=8,
    )
    llm = _get_llm(temperature=0.7, role="profile")
    system = empath_prompt(state["book_title"], state["book_description"])
    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(
            content=f"Create emotional guidelines for: {state['book_title']}"
        ),
    ])
    logger.info("Empath complete (%d chars)", len(response.content))
    return {"emotional_guidelines": response.content}


async def masterclass_node(state: StoryState) -> Dict[str, Any]:
    """Agent 3: generates expert style notes.

    Skips the LLM call if ``expert_styles`` is already set.
    """
    if state.get("expert_styles"):
        logger.info("Masterclass skipped — using stored profile")
        _mark_progress(
            state,
            "masterclass",
            "Expert style guide already available.",
            percent=10,
        )
        return {}

    _mark_progress(
        state,
        "masterclass",
        "Building expert style guide.",
        percent=10,
    )
    llm = _get_llm(temperature=0.7, role="profile")
    system = masterclass_prompt(state["book_title"], state["book_description"])
    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(
            content=f"Create a style guide for: {state['book_title']}"
        ),
    ])
    logger.info("Masterclass complete (%d chars)", len(response.content))
    return {"expert_styles": response.content}


# ---------------------------------------------------------------------------
# Phase 1 — standalone profiling (called directly, NOT via graph)
# ---------------------------------------------------------------------------

async def run_profiling(
    name: str,
    book_title: str,
    book_description: str,
    bio_context: str = "",
    writing_samples: list[str] | None = None,
    expert_style: str = "",
    expert_writing_sample: str = "",
) -> Dict[str, str]:
    """Run all three Phase 1 agents and return their outputs.

    This is called by the ``generate_profile`` view, NOT by the graph.
    It accepts the rich biographical context that the graph nodes don't need.
    """
    llm = _get_llm(temperature=0.7, role="profile")

    samples = writing_samples if writing_samples else None

    # --- Profiler ---
    p_system = profiler_prompt(
        book_title, book_description,
        bio_context=bio_context, writing_samples=samples,
    )
    p_resp = await llm.ainvoke([
        SystemMessage(content=p_system),
        HumanMessage(
            content=f"Analyse everything provided and create a detailed author voice profile for: {name}"
        ),
    ])
    logger.info("Profiling [%s] — Profiler complete (%d chars)", name, len(p_resp.content))

    # --- Empath ---
    e_system = empath_prompt(
        book_title, book_description,
        bio_context=bio_context, writing_samples=samples,
    )
    e_resp = await llm.ainvoke([
        SystemMessage(content=e_system),
        HumanMessage(
            content=f"Analyse everything provided and create emotional guidelines for: {name}"
        ),
    ])
    logger.info("Profiling [%s] — Empath complete (%d chars)", name, len(e_resp.content))

    # --- Masterclass ---
    if expert_style and expert_writing_sample:
        # Enrich the provided expert_style with cues from the writing sample
        enrich_resp = await llm.ainvoke([
            SystemMessage(content=(
                "You are an expert literary style analyst. You have been given an expert's "
                "style notes and a writing sample from that expert. Your job is to produce "
                "a comprehensive, enriched style guide by:\n"
                "1. Preserving everything in the original style notes.\n"
                "2. Extracting additional stylistic cues from the writing sample — sentence "
                "structure, rhythm, tone, vocabulary choices, rhetorical devices, pacing, "
                "use of metaphor, dialogue style, narrative voice, and any other patterns.\n"
                "3. Merging these observations into a single, cohesive style guide.\n\n"
                "Return ONLY the enriched style guide. Do not add commentary."
            )),
            HumanMessage(content=(
                f"## Expert Style Notes\n{expert_style}\n\n"
                f"## Expert Writing Sample\n{expert_writing_sample}"
            )),
        ])
        logger.info("Profiling [%s] — Masterclass enriched expert_style with writing sample (%d chars)", name, len(enrich_resp.content))
        m_content = enrich_resp.content
    elif expert_style:
        logger.info("Profiling [%s] — Masterclass skipped, using provided expert_style (%d chars)", name, len(expert_style))
        m_content = expert_style
    else:
        m_system = masterclass_prompt(
            book_title, book_description,
            bio_context=bio_context, writing_samples=samples,
        )
        m_resp = await llm.ainvoke([
            SystemMessage(content=m_system),
            HumanMessage(
                content=f"Analyse everything provided and create a style guide for: {name}"
            ),
        ])
        logger.info("Profiling [%s] — Masterclass complete (%d chars)", name, len(m_resp.content))
        m_content = m_resp.content

    return {
        "author_profile": p_resp.content,
        "emotional_guidelines": e_resp.content,
        "expert_styles": m_content,
    }


# ---------------------------------------------------------------------------
# Phase 2 — story architect (structured output)
# ---------------------------------------------------------------------------

async def storyteller_node(state: StoryState) -> Dict[str, Any]:
    """Agent 4: plans the entire book.

    Uses ``with_structured_output`` to enforce the ``StoryPlanSchema``
    Pydantic model, guaranteeing valid JSON.
    """
    _mark_progress(
        state,
        "storyteller",
        "Planning the full story architecture.",
        percent=14,
    )
    llm = _get_llm(temperature=0.7, role="storyteller")
    structured_llm = llm.with_structured_output(StoryPlanSchema, method="json_schema")
    system = storyteller_prompt(
        book_title=state["book_title"],
        book_description=state["book_description"],
        num_chapters=state["num_chapters"],
        author_profile=state["author_profile"],
        emotional_guidelines=state["emotional_guidelines"],
        expert_styles=state["expert_styles"],
        webnovel_preferences=state.get("webnovel_preferences") or {},
    )
    plan: StoryPlanSchema = await structured_llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content="Plan the complete book now."),
    ])
    plan_dict = plan.model_dump()
    logger.info(
        "Storyteller complete — %d chapters planned", len(plan_dict["chapters"])
    )
    _mark_progress(
        state,
        "storyteller",
        f"Story plan ready with {len(plan_dict['chapters'])} chapters.",
        percent=20,
    )

    # If the caller didn't specify a batch target, default to writing the
    # whole book in one run (preserves the legacy behaviour).
    total_chapters = len(plan_dict["chapters"])
    existing_target = state.get("target_chapter_index") or 0
    target = existing_target if existing_target > 0 else total_chapters

    return {
        "story_plan": plan_dict,
        "chapters_to_write": plan_dict["chapters"],
        "running_summary": plan_dict.get("initial_summary", ""),
        "current_chapter_index": 0,
        "retry_count": 0,
        "final_chapters": [],
        "target_chapter_index": target,
        "chapter_metadata": [],
        "continuity_ledger": {},
    }


# ---------------------------------------------------------------------------
# Phase 2b — launch chapter planner
# ---------------------------------------------------------------------------

async def launch_chapter_planner_node(state: StoryState) -> Dict[str, Any]:
    """Dedicated Webnovel launch planner for Chapter 1.

    The Storyteller creates the whole serial architecture; this node narrows
    that plan into a conversion-focused brief for the first chapter and launch
    batch. It keeps the main plan intact and stores the launch plan in both
    ``story_plan`` and top-level state for easy prompt injection.
    """
    story_plan = dict(state.get("story_plan") or {})
    if not story_plan:
        logger.warning("LaunchChapterPlanner skipped — no story_plan available")
        return {}

    _mark_progress(
        state,
        "launch_planner",
        "Building Chapter 1 launch plan.",
        percent=22,
    )
    llm = _get_llm(temperature=0.45, role="launch_planner")
    structured_llm = llm.with_structured_output(
        LaunchChapterPlan,
        method="json_schema",
    )
    system = launch_chapter_planner_prompt(
        book_title=state["book_title"],
        book_description=state["book_description"],
        story_plan=story_plan,
        webnovel_preferences=state.get("webnovel_preferences") or {},
    )
    launch_plan: LaunchChapterPlan = await structured_llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(
            content="Build the Chapter 1 launch-conversion plan now."
        ),
    ])
    launch_plan_dict = launch_plan.model_dump()

    story_plan["launch_chapter_plan"] = launch_plan_dict

    # Keep chapter 1's plan aligned with the dedicated launch brief without
    # overwriting the rest of the chapter architecture.
    chapters_to_write = [
        dict(chapter) for chapter in (state.get("chapters_to_write") or [])
    ]
    if chapters_to_write:
        first = dict(chapters_to_write[0])
        if launch_plan_dict.get("first_200_words_hook"):
            first["opening_hook"] = launch_plan_dict["first_200_words_hook"]
        if launch_plan_dict.get("chapter_one_progression_reward"):
            first["progression_reward"] = launch_plan_dict[
                "chapter_one_progression_reward"
            ]
        if launch_plan_dict.get("chapter_one_cliffhanger"):
            first["cliffhanger"] = launch_plan_dict["chapter_one_cliffhanger"]
        chapters_to_write[0] = first
        story_plan["chapters"] = chapters_to_write

    logger.info("LaunchChapterPlanner complete for story %s", state.get("story_id"))
    _mark_progress(
        state,
        "launch_planner",
        "Chapter 1 launch plan ready.",
        percent=25,
    )
    return {
        "story_plan": story_plan,
        "chapters_to_write": chapters_to_write,
        "launch_chapter_plan": launch_plan_dict,
    }


# ---------------------------------------------------------------------------
# Phase 3 — chapter writer (dynamic prompt injection)
# ---------------------------------------------------------------------------

async def writer_node(state: StoryState) -> Dict[str, Any]:
    """Agent 5: writes one chapter.

    The system prompt is *dynamically constructed* by injecting outputs from
    agents 1-3 and the running summary for context-window management.  On
    resume (chapter >= 2), the continuity ledger and the summaries of the
    last two chapters are also injected so the Writer stays consistent.
    """
    llm = _get_llm(temperature=0.8, role="writer")
    idx = state["current_chapter_index"]
    chapter_plan = state["chapters_to_write"][idx]
    _mark_progress(
        state,
        "writer",
        f"Writing Chapter {idx + 1}: {chapter_plan.get('title', '')}",
        chapter_number=idx + 1,
        phase_fraction=0.15,
    )

    # Closing excerpt of the previous chapter — prefer the richer 800-char
    # excerpt stored in chapter_metadata when available; fall back to slicing
    # the raw text for the legacy flow.
    previous_ending = ""
    chapter_metadata = state.get("chapter_metadata") or []
    if chapter_metadata:
        previous_ending = chapter_metadata[-1].get("closing_excerpt", "")
    if not previous_ending and state.get("final_chapters"):
        previous_ending = state["final_chapters"][-1][-800:]

    # Continuity anchors — empty on chapter 1, populated from chapter 2 onwards.
    continuity_ledger = state.get("continuity_ledger") or {}
    recent_window = int(getattr(django_settings, "RAG_RECENT_WINDOW", 2) or 0)
    recent_summaries = (
        chapter_metadata[-recent_window:] if recent_window and chapter_metadata else []
    )

    # RAG: retrieve the K most semantically-relevant earlier chapters for
    # the upcoming chapter. The recency window above is excluded so the
    # writer doesn't see those chapters twice. Fails soft to [].
    rag_top_k = int(getattr(django_settings, "RAG_TOP_K", 0) or 0)
    relevant_past_summaries: list[Dict[str, Any]] = []
    if rag_top_k > 0 and chapter_metadata:
        try:
            relevant_past_summaries = await select_relevant_summaries(
                build_retrieval_query(chapter_plan),
                chapter_metadata,
                k=rag_top_k,
                exclude_recent=recent_window,
            )
        except Exception:
            logger.exception(
                "RAG retrieval failed for chapter %d — falling back to recency only",
                idx + 1,
            )

    launch_plan = {}
    if idx == 0:
        launch_plan = (
            state.get("launch_chapter_plan")
            or (state.get("story_plan") or {}).get("launch_chapter_plan")
            or {}
        )

    system = writer_prompt(
        chapter_plan=chapter_plan,
        author_profile=state["author_profile"],
        emotional_guidelines=state["emotional_guidelines"],
        expert_styles=state["expert_styles"],
        running_summary=state["running_summary"],
        previous_chapter_ending=previous_ending,
        min_words=django_settings.MIN_CHAPTER_WORDS,
        continuity_ledger=continuity_ledger,
        recent_chapter_summaries=recent_summaries,
        relevant_past_summaries=relevant_past_summaries,
        launch_chapter_plan=launch_plan,
    )
    user_message = HumanMessage(
        content=f"Write Chapter {idx + 1}: {chapter_plan.get('title', '')}"
    )
    response = await llm.ainvoke([SystemMessage(content=system), user_message])

    # Reasoning-capable Together models can burn the entire max_tokens budget
    # on hidden reasoning and return content="" with finish_reason="length".
    # Retry once with a smaller token budget (more room for visible output)
    # before handing the empty draft to the reviewer.
    if not (response.content or "").strip():
        logger.warning(
            "Writer returned empty content for chapter %d — metadata=%s. Retrying once.",
            idx + 1,
            getattr(response, "response_metadata", {}),
        )
        retry_max_tokens = _max_tokens_for_role("writer")
        retry_max_tokens = int(retry_max_tokens * 0.7) if retry_max_tokens else None
        retry_llm = ChatTogether(
            api_key=django_settings.TOGETHER_API_KEY,
            model=_model_for_role("writer"),
            temperature=0.9,
            max_tokens=retry_max_tokens,
            max_retries=3,
            timeout=120,
        )
        response = await retry_llm.ainvoke([SystemMessage(content=system), user_message])
        if not (response.content or "").strip():
            logger.warning(
                "Writer retry also empty for chapter %d — metadata=%s",
                idx + 1,
                getattr(response, "response_metadata", {}),
            )

    logger.info(
        "Writer complete — chapter %d (%d chars)", idx + 1, len(response.content or "")
    )
    return {"current_draft": response.content or ""}


# ---------------------------------------------------------------------------
# Phase 4 — reviewer (structured output)
# ---------------------------------------------------------------------------

async def reviewer_node(state: StoryState) -> Dict[str, Any]:
    """Agent 7: evaluates the current draft.

    First performs a programmatic word-count check — if the chapter is under
    the minimum, it auto-rejects without spending tokens on the reviewer LLM.

    Returns a ``ReviewVerdict`` (approve / revise) and increments
    ``retry_count``.  The transient ``_review_status`` field is read by the
    conditional-edge router in ``graph.py``.
    """
    idx = state["current_chapter_index"]
    new_retry = state["retry_count"] + 1
    min_words = django_settings.MIN_CHAPTER_WORDS
    _mark_progress(
        state,
        "reviewer",
        f"Reviewing Chapter {idx + 1}.",
        chapter_number=idx + 1,
        phase_fraction=0.45,
        extra={"retry_count": new_retry},
    )

    # --- Programmatic word-count gate (saves an LLM call) ---
    word_count = len(state["current_draft"].split())
    if word_count < min_words:
        deficit = min_words - word_count
        feedback = (
            f"AUTOMATIC REJECTION: Chapter is only {word_count} words — "
            f"{deficit} words short of the {min_words}-word minimum. "
            "You MUST expand the chapter significantly. Add: "
            "1) Full dramatised scenes with sensory details instead of summaries. "
            "2) Extended dialogue exchanges (4+ lines) with action beats. "
            "3) Character interiority — inner thoughts, memories, emotional reactions. "
            "4) Environmental descriptions and atmosphere-building passages. "
            "5) Transitional moments that show characters moving through the world."
        )
        logger.warning(
            "Chapter %d auto-rejected: %d words (min %d)",
            idx + 1, word_count, min_words,
        )
        return {
            "review_feedback": feedback,
            "retry_count": new_retry,
            "_review_status": "revise",
        }

    # --- Full LLM review (only if word count passes) ---
    llm = _get_llm(temperature=0.3, role="reviewer")
    structured_llm = llm.with_structured_output(ReviewVerdict, method="json_schema")
    chapter_plan = state["chapters_to_write"][idx]
    recent_metadata = (state.get("chapter_metadata") or [])[-3:]
    launch_plan = {}
    if idx == 0:
        launch_plan = (
            state.get("launch_chapter_plan")
            or (state.get("story_plan") or {}).get("launch_chapter_plan")
            or {}
        )
    system = reviewer_prompt(
        chapter_plan,
        recent_chapter_metadata=recent_metadata,
        launch_chapter_plan=launch_plan,
    )

    verdict: ReviewVerdict = await structured_llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=state["current_draft"]),
    ])
    logger.info(
        "Reviewer verdict for chapter %d: %s (retry %d)",
        idx + 1,
        verdict.status,
        new_retry,
    )
    return {
        "review_feedback": verdict.feedback,
        "retry_count": new_retry,
        "_review_status": verdict.status,  # transient — read by router
    }


# ---------------------------------------------------------------------------
# Phase 5 — perfectionist / humaniser
# ---------------------------------------------------------------------------

async def perfectionist_node(state: StoryState) -> Dict[str, Any]:
    """Agent 6: rewrites the chapter based on reviewer feedback."""
    llm = _get_llm(temperature=0.6, role="perfectionist")
    idx = state["current_chapter_index"]
    chapter_plan = state["chapters_to_write"][idx]
    _mark_progress(
        state,
        "perfectionist",
        f"Revising Chapter {idx + 1} after review.",
        chapter_number=idx + 1,
        phase_fraction=0.65,
        extra={"retry_count": state.get("retry_count", 0)},
    )
    launch_plan = {}
    if idx == 0:
        launch_plan = (
            state.get("launch_chapter_plan")
            or (state.get("story_plan") or {}).get("launch_chapter_plan")
            or {}
        )
    system = perfectionist_prompt(
        current_draft=state["current_draft"],
        review_feedback=state["review_feedback"],
        chapter_plan=chapter_plan,
        launch_chapter_plan=launch_plan,
    )
    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content="Rewrite the chapter addressing all feedback."),
    ])
    logger.info(
        "Perfectionist rewrite for chapter %d (%d chars)",
        idx + 1,
        len(response.content),
    )
    return {"current_draft": response.content}


# ---------------------------------------------------------------------------
# Chapter acceptance helper
# ---------------------------------------------------------------------------

async def accept_chapter_node(state: StoryState) -> Dict[str, Any]:
    """Append the approved draft, update the running summary, advance index.

    Also persists a ``ChapterMetadata`` record (title, word count, summary,
    opening/closing excerpts) so the continuation flow has a rich recency
    anchor without having to re-read the full previous chapter text.
    """
    idx = state["current_chapter_index"]
    draft = state["current_draft"] or ""
    retry_count = state.get("retry_count", 0)
    chapter_plan = state["chapters_to_write"][idx]

    # An empty draft is never accepted on its own merit. If we still have
    # retries left, bounce it back through the perfectionist. If retries are
    # exhausted, fail the run with a clear error rather than writing a
    # zero-word chapter into the manuscript.
    if not draft.strip():
        if retry_count < 3:
            logger.warning(
                "Chapter %d draft is empty — sending back through perfectionist (retry %d/3)",
                idx + 1,
                retry_count + 1,
            )
            return {
                "_review_status": "revise",
                "review_feedback": (
                    "Writer produced an empty chapter. Rewrite the chapter from scratch "
                    "using the chapter plan, continuity ledger, and previous chapter "
                    "summaries as context. Aim for the full minimum word count."
                ),
                "retry_count": retry_count + 1,
            }
        error_msg = (
            f"Chapter {idx + 1} could not be drafted after {retry_count} retries — "
            "writer kept returning empty content."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    _mark_progress(
        state,
        "accept_chapter",
        f"Accepting and summarising Chapter {idx + 1}.",
        chapter_number=idx + 1,
        phase_fraction=0.82,
    )

    # Deterministic fallback summary built from the chapter plan — used when
    # the summary LLM call fails. Reuses fields already required below for
    # chapter_metadata, so no new prompt engineering is needed.
    fallback_summary = (
        f"Chapter {idx + 1}: {chapter_plan.get('title', '') or 'Untitled'}. "
        f"Key events: "
        f"{', '.join(chapter_plan.get('key_events', []) or []) or 'see chapter plan'}."
    )

    try:
        llm = _get_llm(temperature=0.3, role="summary")
        summary_resp = await llm.ainvoke([
            SystemMessage(
                content="Summarise the following chapter in 2-3 sentences.  "
                "Focus on plot progression and character development.  "
                "Be concise — this will be used as context for writing "
                "subsequent chapters."
            ),
            HumanMessage(content=draft),
        ])
        chapter_summary = (summary_resp.content or "").strip() or fallback_summary
    except Exception:
        logger.exception(
            "Summary LLM failed for chapter %d — using deterministic fallback",
            idx + 1,
        )
        chapter_summary = fallback_summary

    # Embed the summary so the next Writer call can retrieve it via RAG.
    # Failure is non-fatal — the chapter still gets accepted, just without
    # a vector. select_relevant_summaries skips entries that lack one.
    summary_embedding = await embed_text(chapter_summary) or []

    # `running_summary` is no longer extended chapter-by-chapter (that string
    # used to grow linearly and would blow the Writer prompt past ~30
    # chapters). It now stays as the immutable book synopsis set by the
    # Storyteller, and per-chapter context comes from RAG + the recency
    # window in writer_node.
    updated_summary = state["running_summary"]

    launch_plan = (
        state.get("launch_chapter_plan")
        or (state.get("story_plan") or {}).get("launch_chapter_plan")
        or {}
    )
    new_metadata_entry = {
        "chapter_number": idx + 1,
        "title": chapter_plan.get("title", ""),
        "word_count": len(draft.split()),
        "summary": chapter_summary,
        "characters_appeared": list(chapter_plan.get("characters_involved", []) or []),
        "key_events_delivered": list(chapter_plan.get("key_events", []) or []),
        "launch_plan_applied": idx == 0 and bool(launch_plan),
        "first_200_word_goal": launch_plan.get("first_200_words_hook", "")
        if idx == 0
        else "",
        "comment_magnet_question": launch_plan.get("comment_magnet_question", "")
        if idx == 0
        else "",
        "opening_hook": chapter_plan.get("opening_hook", ""),
        "progression_reward": chapter_plan.get("progression_reward", ""),
        "new_question_raised": chapter_plan.get("new_question_raised", ""),
        "cliffhanger": chapter_plan.get("cliffhanger", ""),
        "reader_emotion_target": chapter_plan.get("reader_emotion_target", ""),
        "tags_served": list(chapter_plan.get("tags_served", []) or []),
        "comment_prompt": chapter_plan.get("comment_prompt", ""),
        "power_stone_pitch": chapter_plan.get("power_stone_pitch", ""),
        "filler_risk": chapter_plan.get("filler_risk", ""),
        "opening_excerpt": draft[:300],
        "closing_excerpt": draft[-800:] if len(draft) > 800 else draft,
        "summary_embedding": summary_embedding,
    }
    chapter_metadata = list(state.get("chapter_metadata") or [])
    chapter_metadata.append(new_metadata_entry)

    logger.info("Chapter %d accepted and summarised", idx + 1)
    final_chapters = state["final_chapters"] + [draft]
    progress_state = {
        **state,
        "final_chapters": final_chapters,
        "current_chapter_index": idx + 1,
    }
    _mark_progress(
        progress_state,
        "chapter_complete",
        f"Chapter {idx + 1} accepted.",
        chapter_number=idx + 1,
        phase_fraction=0.0,
    )
    return {
        "final_chapters": final_chapters,
        "running_summary": updated_summary,
        "current_chapter_index": idx + 1,
        "retry_count": 0,
        "current_draft": "",
        "review_feedback": "",
        "chapter_metadata": chapter_metadata,
    }


# ---------------------------------------------------------------------------
# Post-chapter — continuity extractor
# ---------------------------------------------------------------------------

async def continuity_extractor_node(state: StoryState) -> Dict[str, Any]:
    """Refresh the ``continuity_ledger`` after a chapter is accepted.

    Runs after ``accept_chapter_node``. Takes the full text of the chapter
    just added (``final_chapters[-1]``) and the existing ledger, and
    produces an updated ledger using structured output.  This is what makes
    the ``/continue/`` flow safe against long pauses — the ledger is the
    ground truth for character locations, open plot threads, named
    entities, etc.
    """
    final_chapters = state.get("final_chapters") or []
    if not final_chapters:
        # Shouldn't happen — accept_chapter_node always runs first — but
        # guard anyway to avoid crashing the graph.
        logger.warning("continuity_extractor skipped — no accepted chapters yet")
        return {}

    # The chapter we just accepted is the last one in final_chapters, and
    # current_chapter_index was already advanced past it by accept_chapter_node.
    last_chapter_idx = state["current_chapter_index"] - 1
    chapter_text = final_chapters[-1]
    chapter_plan = state["chapters_to_write"][last_chapter_idx]
    existing_ledger = state.get("continuity_ledger") or {}
    _mark_progress(
        state,
        "continuity",
        f"Refreshing continuity after Chapter {last_chapter_idx + 1}.",
        chapter_number=last_chapter_idx + 1,
        phase_fraction=0.95,
    )

    llm = _get_llm(temperature=0.2, role="continuity")
    structured_llm = llm.with_structured_output(ContinuityLedger, method="json_schema")
    system = continuity_extractor_prompt(
        chapter_number=last_chapter_idx + 1,
        existing_ledger=existing_ledger,
        chapter_plan=chapter_plan,
    )

    try:
        updated: ContinuityLedger = await structured_llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=chapter_text),
        ])
        logger.info(
            "Continuity ledger refreshed after chapter %d (%d characters tracked)",
            last_chapter_idx + 1,
            len(updated.characters),
        )
        return {"continuity_ledger": updated.model_dump()}
    except Exception:
        # A failure to refresh the ledger must NOT abort the whole book —
        # worst case, the next chapter falls back to the previous ledger.
        logger.exception(
            "continuity_extractor failed for chapter %d — keeping previous ledger",
            last_chapter_idx + 1,
        )
        return {}


# ---------------------------------------------------------------------------
# Phase 6 — publisher / compiler
# ---------------------------------------------------------------------------

def _format_chapter_digest_entry(meta: Dict[str, Any]) -> str:
    """Render one chapter_metadata record into a compact, audit-ready block.

    Uses the planned hook/cliffhanger fields plus short opening/closing
    excerpts so the Publisher can audit serial retention without seeing the
    full chapter text. Excerpts are clipped to keep the digest bounded as
    the book grows past ~30 chapters.
    """
    excerpt_cap = 200
    opening = (meta.get("opening_excerpt") or "")[:excerpt_cap]
    closing_full = meta.get("closing_excerpt") or ""
    closing = closing_full[-excerpt_cap:] if len(closing_full) > excerpt_cap else closing_full
    key_events = meta.get("key_events_delivered") or []
    lines = [
        f"[Ch.{meta.get('chapter_number', '?')}] {meta.get('title', '')} "
        f"({meta.get('word_count', 0)} words)",
        f"Summary: {meta.get('summary', '')}",
    ]
    if meta.get("opening_hook"):
        lines.append(f"Planned opening hook: {meta['opening_hook']}")
    if meta.get("cliffhanger"):
        lines.append(f"Planned cliffhanger: {meta['cliffhanger']}")
    if meta.get("progression_reward"):
        lines.append(f"Progression reward: {meta['progression_reward']}")
    if key_events:
        lines.append(f"Key events: {'; '.join(key_events)}")
    if opening:
        lines.append(f"Opening excerpt: {opening}")
    if closing:
        lines.append(f"Closing excerpt: {closing}")
    return "\n".join(lines)


def _build_publisher_payload(
    final_chapters: List[str],
    chapter_metadata: List[Dict[str, Any]],
) -> str:
    """Assemble the Publisher's input: Ch.1 in full + per-chapter digest.

    Falls back to opening/closing excerpts of each chapter when metadata is
    missing (shouldn't happen in normal flow, but keeps the publisher safe
    against partial state from older runs).
    """
    parts: List[str] = []
    if final_chapters:
        parts.append("=== CHAPTER 1 (FULL TEXT) ===\n" + final_chapters[0])

    if chapter_metadata:
        digest = "\n\n".join(
            _format_chapter_digest_entry(m) for m in chapter_metadata
        )
        parts.append("=== CHAPTER DIGEST ===\n" + digest)
    elif len(final_chapters) > 1:
        fallback_blocks = []
        for i, ch in enumerate(final_chapters[1:], start=2):
            opening = ch[:300]
            closing = ch[-500:] if len(ch) > 500 else ch
            fallback_blocks.append(
                f"[Ch.{i}] (no metadata)\n"
                f"Opening: {opening}\n"
                f"Closing: {closing}"
            )
        parts.append("=== CHAPTER DIGEST (fallback) ===\n" + "\n\n".join(fallback_blocks))

    return "\n\n".join(parts)


async def publisher_node(state: StoryState) -> Dict[str, Any]:
    """Agent 8: compiles the final manuscript and generates sequel hooks.

    Sends a bounded digest (Ch.1 full + per-chapter metadata) to the LLM
    instead of joining every chapter's text — that join scales linearly with
    chapter count and blows past the model context window past ~30 chapters.
    """
    _mark_progress(
        state,
        "publisher",
        "Compiling final manuscript and publishing notes.",
        percent=96,
    )
    llm = _get_llm(temperature=0.7, role="publisher")
    system = publisher_prompt(state["book_title"], state["story_plan"])

    payload = _build_publisher_payload(
        state["final_chapters"],
        state.get("chapter_metadata") or [],
    )
    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(
            content=(
                "Below is Chapter 1 in full plus a structured digest of every "
                "chapter. Produce the blurb, publishing package, launch assets, "
                "Chapter 1 audit, retention audit, sequel hooks, and editorial "
                "notes per your instructions.\n\n"
                + payload
            )
        ),
    ])

    manuscript = {
        "title": state["book_title"],
        "story_plan": state["story_plan"],
        "webnovel_strategy": state["story_plan"].get("webnovel_strategy", {}),
        "launch_chapter_plan": state["story_plan"].get("launch_chapter_plan", {}),
        "serial_arcs": state["story_plan"].get("serial_arcs", []),
        "release_plan": state["story_plan"].get("release_plan", []),
        "retention_strategy": state["story_plan"].get("retention_strategy", []),
        "chapters": [
            {"chapter_number": i + 1, "text": ch}
            for i, ch in enumerate(state["final_chapters"])
        ],
        "publisher_notes": response.content,
        "running_summary": state["running_summary"],
    }

    logger.info("Publisher complete — manuscript compiled")
    return {
        "final_manuscript": manuscript,
        "status": "completed",
    }
