"""LangGraph StateGraph definition for the multi-agent story pipeline.

Two compiled graphs are exposed:

- ``story_graph``    — full pipeline (Phase 1 → 6). Used by ``create_story``.
- ``continue_graph`` — resumes an existing story at the Writer. Used by
  ``continue_story_view`` after the initial batch has finished.

Graph flow (full pipeline)::

    START ─┬─> profiler    ─┐
           ├─> empath      ─┼─> storyteller ─> launch_planner ─> writer ─> reviewer
           └─> masterclass ─┘                    ^           │
                                                 │     ┌─────┴─────┐
                                                 │  approve/      revise &
                                                 │  max_retry      retry<3
                                                 │     │              │
                                                 │     v              v
                                           accept_chapter     perfectionist
                                                 │                    │
                                                 v              (back to
                                       continuity_extractor      reviewer)
                                                 │
                                       ┌─────────┼─────────┐
                                       │         │         │
                                    writer   publisher    END
                                   (batch     (book      (batch done,
                                   open)      done)      book open)
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from .agents import (
    accept_chapter_node,
    continuity_extractor_node,
    empath_node,
    launch_chapter_planner_node,
    masterclass_node,
    perfectionist_node,
    profiler_node,
    publisher_node,
    reviewer_node,
    storyteller_node,
    writer_node,
)
from .state import StoryState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional-edge routers
# ---------------------------------------------------------------------------

def _review_router(state: StoryState) -> str:
    """Route after the Reviewer node.

    - ``approve`` OR ``retry_count >= 3`` → accept the chapter.
    - ``revise`` AND ``retry_count < 3``  → send to the Perfectionist.

    Default is "revise" — a missing status should err on the side of
    rewriting rather than silently accepting whatever's in current_draft.
    """
    status = state.get("_review_status", "revise")
    retry_count = state.get("retry_count", 0)
    logger.info(
        "review_router: status=%s retry=%d chapter=%d",
        status,
        retry_count,
        state["current_chapter_index"] + 1,
    )
    if status == "approve" or retry_count >= 3:
        if retry_count >= 3 and status != "approve":
            logger.warning(
                "Max retries reached for chapter %d — accepting as-is",
                state["current_chapter_index"] + 1,
            )
        return "accept_chapter"
    return "perfectionist"


def _chapter_router(state: StoryState) -> str:
    """Route after the continuity extractor has refreshed the ledger.

    Three outcomes, in priority order:

    - ``publisher`` — every planned chapter is written; compile the book.
    - ``batch_done`` — this batch's target has been reached but the book
      still has more chapters; park the run and wait for ``/continue/``.
    - ``writer`` — still inside the current batch; draft the next chapter.
    """
    total = len(state["chapters_to_write"])
    idx = state["current_chapter_index"]
    target = state.get("target_chapter_index") or total

    if idx >= total:
        logger.info("All %d chapters complete — moving to publisher", total)
        return "publisher"
    if idx >= target:
        logger.info(
            "Batch target reached (%d / %d) — parking for /continue/",
            idx, total,
        )
        return "batch_done"
    logger.info(
        "Advancing to chapter %d / %d (batch target %d)",
        idx + 1, total, target,
    )
    return "writer"


# ---------------------------------------------------------------------------
# Shared edges — the writer → reviewer → perfectionist → accept → extractor
# loop is identical in both the full and continue graphs.
# ---------------------------------------------------------------------------

def _add_chapter_loop_edges(graph: StateGraph) -> None:
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("perfectionist", perfectionist_node)
    graph.add_node("accept_chapter", accept_chapter_node)
    graph.add_node("continuity_extractor", continuity_extractor_node)
    graph.add_node("publisher", publisher_node)

    graph.add_edge("writer", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        _review_router,
        {"accept_chapter": "accept_chapter", "perfectionist": "perfectionist"},
    )
    graph.add_edge("perfectionist", "reviewer")

    # Every accepted chapter flows through the continuity extractor before
    # the router decides what to do next.
    graph.add_edge("accept_chapter", "continuity_extractor")
    graph.add_conditional_edges(
        "continuity_extractor",
        _chapter_router,
        {"writer": "writer", "publisher": "publisher", "batch_done": END},
    )

    graph.add_edge("publisher", END)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_full_graph() -> StateGraph:
    """Full pipeline: Phase 1 (parallel) → Storyteller → chapter loop."""
    graph = StateGraph(StoryState)

    graph.add_node("profiler", profiler_node)
    graph.add_node("empath", empath_node)
    graph.add_node("masterclass", masterclass_node)
    graph.add_node("storyteller", storyteller_node)
    graph.add_node("launch_planner", launch_chapter_planner_node)

    # --- Phase 1: parallel fan-out from START ---
    graph.add_edge(START, "profiler")
    graph.add_edge(START, "empath")
    graph.add_edge(START, "masterclass")

    # --- Phase 1 → Phase 2: fan-in to storyteller ---
    graph.add_edge("profiler", "storyteller")
    graph.add_edge("empath", "storyteller")
    graph.add_edge("masterclass", "storyteller")

    # --- Phase 2 → Phase 2b → Phase 3: storyteller → launch planner → writer ---
    graph.add_edge("storyteller", "launch_planner")
    graph.add_edge("launch_planner", "writer")

    _add_chapter_loop_edges(graph)

    return graph.compile()


def build_continue_graph() -> StateGraph:
    """Resume pipeline: skips Phase 1-2 and starts directly at the Writer.

    Phase 1-2 outputs (author_profile, emotional_guidelines, expert_styles,
    story_plan, chapters_to_write, continuity_ledger, chapter_metadata) are
    expected to be present in the state loaded from MongoDB.
    """
    graph = StateGraph(StoryState)

    _add_chapter_loop_edges(graph)

    graph.add_edge(START, "writer")

    return graph.compile()


# Module-level compiled graphs (singletons).
story_graph = build_full_graph()
continue_graph = build_continue_graph()
