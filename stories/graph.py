"""LangGraph StateGraph definition for the multi-agent story pipeline.

Graph flow:

    START ─┬─> profiler    ─┐
           ├─> empath      ─┼─> storyteller ─> writer ─> reviewer
           └─> masterclass ─┘                    ^           │
                                                 │     ┌─────┴─────┐
                                                 │  approve/      revise &
                                                 │  max_retry      retry<3
                                                 │     │              │
                                                 │     v              v
                                           accept_chapter     perfectionist
                                                 │                    │
                                           ┌─────┴─────┐        (back to
                                       more_chs    done         reviewer)
                                           │          │
                                           v          v
                                         writer    publisher ─> END
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from .agents import (
    accept_chapter_node,
    empath_node,
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
    """
    status = state.get("_review_status", "approve")
    if status == "approve" or state["retry_count"] >= 3:
        if state["retry_count"] >= 3 and status != "approve":
            logger.warning(
                "Max retries reached for chapter %d — accepting as-is",
                state["current_chapter_index"] + 1,
            )
        return "accept_chapter"
    return "perfectionist"


def _chapter_router(state: StoryState) -> str:
    """Route after accepting a chapter.

    - More chapters remaining → loop back to the Writer.
    - All chapters done    → proceed to the Publisher.
    """
    if state["current_chapter_index"] < len(state["chapters_to_write"]):
        logger.info(
            "Advancing to chapter %d / %d",
            state["current_chapter_index"] + 1,
            len(state["chapters_to_write"]),
        )
        return "writer"
    logger.info("All chapters complete — moving to publisher")
    return "publisher"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the story-writing StateGraph."""

    graph = StateGraph(StoryState)

    # --- Register nodes ---
    graph.add_node("profiler", profiler_node)
    graph.add_node("empath", empath_node)
    graph.add_node("masterclass", masterclass_node)
    graph.add_node("storyteller", storyteller_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("perfectionist", perfectionist_node)
    graph.add_node("accept_chapter", accept_chapter_node)
    graph.add_node("publisher", publisher_node)

    # --- Phase 1: parallel fan-out from START ---
    graph.add_edge(START, "profiler")
    graph.add_edge(START, "empath")
    graph.add_edge(START, "masterclass")

    # --- Phase 1 → Phase 2: fan-in to storyteller ---
    graph.add_edge("profiler", "storyteller")
    graph.add_edge("empath", "storyteller")
    graph.add_edge("masterclass", "storyteller")

    # --- Phase 2 → Phase 3: storyteller → writer ---
    graph.add_edge("storyteller", "writer")

    # --- Phase 3 → Phase 4: writer → reviewer ---
    graph.add_edge("writer", "reviewer")

    # --- Phase 4: conditional routing from reviewer ---
    graph.add_conditional_edges(
        "reviewer",
        _review_router,
        {"accept_chapter": "accept_chapter", "perfectionist": "perfectionist"},
    )

    # --- Phase 5: perfectionist loops back to reviewer ---
    graph.add_edge("perfectionist", "reviewer")

    # --- After accepting: next chapter or publish ---
    graph.add_conditional_edges(
        "accept_chapter",
        _chapter_router,
        {"writer": "writer", "publisher": "publisher"},
    )

    # --- Phase 6: publisher → END ---
    graph.add_edge("publisher", END)

    return graph.compile()


# Module-level compiled graph (singleton).
story_graph = build_graph()
