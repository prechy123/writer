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
from typing import Any, Dict

from django.conf import settings as django_settings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from .prompts import (
    continuity_extractor_prompt,
    empath_prompt,
    masterclass_prompt,
    perfectionist_prompt,
    profiler_prompt,
    publisher_prompt,
    reviewer_prompt,
    storyteller_prompt,
    writer_prompt,
)
from .schemas import ContinuityLedger, ReviewVerdict, StoryPlanSchema
from .state import StoryState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_llm(temperature: float = 0.7, max_retries: int = 3) -> ChatGroq:
    return ChatGroq(
        api_key=django_settings.GROQ_API_KEY,
        model=django_settings.GROQ_MODEL,
        temperature=temperature,
        max_retries=max_retries,
        request_timeout=120,
    )


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
        return {}

    llm = _get_llm(temperature=0.7)
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
        return {}

    llm = _get_llm(temperature=0.7)
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
        return {}

    llm = _get_llm(temperature=0.7)
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
    llm = _get_llm(temperature=0.7)

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
    llm = _get_llm(temperature=0.7)
    structured_llm = llm.with_structured_output(StoryPlanSchema, method="json_schema")
    system = storyteller_prompt(
        book_title=state["book_title"],
        book_description=state["book_description"],
        num_chapters=state["num_chapters"],
        author_profile=state["author_profile"],
        emotional_guidelines=state["emotional_guidelines"],
        expert_styles=state["expert_styles"],
    )
    plan: StoryPlanSchema = await structured_llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content="Plan the complete book now."),
    ])
    plan_dict = plan.model_dump()
    logger.info(
        "Storyteller complete — %d chapters planned", len(plan_dict["chapters"])
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
# Phase 3 — chapter writer (dynamic prompt injection)
# ---------------------------------------------------------------------------

async def writer_node(state: StoryState) -> Dict[str, Any]:
    """Agent 5: writes one chapter.

    The system prompt is *dynamically constructed* by injecting outputs from
    agents 1-3 and the running summary for context-window management.  On
    resume (chapter >= 2), the continuity ledger and the summaries of the
    last two chapters are also injected so the Writer stays consistent.
    """
    llm = _get_llm(temperature=0.8)
    idx = state["current_chapter_index"]
    chapter_plan = state["chapters_to_write"][idx]

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
    recent_summaries = chapter_metadata[-2:] if chapter_metadata else []

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
    )
    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(
            content=f"Write Chapter {idx + 1}: {chapter_plan.get('title', '')}"
        ),
    ])
    logger.info(
        "Writer complete — chapter %d (%d chars)", idx + 1, len(response.content)
    )
    return {"current_draft": response.content}


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
    llm = _get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(ReviewVerdict, method="json_schema")
    chapter_plan = state["chapters_to_write"][idx]
    system = reviewer_prompt(chapter_plan)

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
    llm = _get_llm(temperature=0.6)
    idx = state["current_chapter_index"]
    chapter_plan = state["chapters_to_write"][idx]
    system = perfectionist_prompt(
        current_draft=state["current_draft"],
        review_feedback=state["review_feedback"],
        chapter_plan=chapter_plan,
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
    draft = state["current_draft"]

    llm = _get_llm(temperature=0.3)
    summary_resp = await llm.ainvoke([
        SystemMessage(
            content="Summarise the following chapter in 2-3 sentences.  "
            "Focus on plot progression and character development.  "
            "Be concise — this will be used as context for writing "
            "subsequent chapters."
        ),
        HumanMessage(content=draft),
    ])
    chapter_summary = summary_resp.content

    updated_summary = (
        state["running_summary"]
        + f"\n\nChapter {idx + 1}: {chapter_summary}"
    )

    chapter_plan = state["chapters_to_write"][idx]
    new_metadata_entry = {
        "chapter_number": idx + 1,
        "title": chapter_plan.get("title", ""),
        "word_count": len(draft.split()),
        "summary": chapter_summary,
        "characters_appeared": list(chapter_plan.get("characters_involved", []) or []),
        "key_events_delivered": list(chapter_plan.get("key_events", []) or []),
        "opening_excerpt": draft[:300],
        "closing_excerpt": draft[-800:] if len(draft) > 800 else draft,
    }
    chapter_metadata = list(state.get("chapter_metadata") or [])
    chapter_metadata.append(new_metadata_entry)

    logger.info("Chapter %d accepted and summarised", idx + 1)
    return {
        "final_chapters": state["final_chapters"] + [draft],
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

    llm = _get_llm(temperature=0.2)
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

async def publisher_node(state: StoryState) -> Dict[str, Any]:
    """Agent 8: compiles the final manuscript and generates sequel hooks."""
    llm = _get_llm(temperature=0.7)
    system = publisher_prompt(state["book_title"], state["story_plan"])

    all_chapters = "\n\n--- CHAPTER BREAK ---\n\n".join(state["final_chapters"])
    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(
            content=(
                "Here are all the completed chapters.  "
                "Produce the blurb, sequel hooks and editorial notes.\n\n"
                + all_chapters
            )
        ),
    ])

    manuscript = {
        "title": state["book_title"],
        "story_plan": state["story_plan"],
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
