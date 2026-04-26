from typing import Any, Dict, List, TypedDict


class StoryState(TypedDict, total=False):
    # --- User inputs ---
    book_title: str
    book_description: str
    num_chapters: int
    webnovel_preferences: Dict[str, Any]

    # --- Phase 1: Context (populated by Profiler, Empath, Masterclass) ---
    author_profile: str
    emotional_guidelines: str
    expert_styles: str

    # --- Phase 2: Story Architecture (populated by Storyteller) ---
    story_plan: Dict[str, Any]
    chapters_to_write: List[Dict]
    launch_chapter_plan: Dict[str, Any]

    # --- Phase 3-5: Drafting & Review loop ---
    current_chapter_index: int
    current_draft: str
    review_feedback: str
    retry_count: int
    running_summary: str
    # Transient verdict from reviewer_node, read by graph._review_router.
    # Declared explicitly so LangGraph's channel system propagates it.
    _review_status: str

    # --- Batch control ---
    # Chapter loop stops when current_chapter_index >= target_chapter_index.
    # Set at story creation (initial_chapters) and bumped by the /continue/ endpoint.
    target_chapter_index: int
    progress: Dict[str, Any]

    # --- Chapter accumulator ---
    final_chapters: List[str]

    # --- Rich continuity memory (persisted so continuation has strong anchors) ---
    # Appended after each accepted chapter (see schemas.ChapterMetadata).
    chapter_metadata: List[Dict[str, Any]]
    # Living world/character/plot state, refreshed by continuity_extractor_node
    # (see schemas.ContinuityLedger).
    continuity_ledger: Dict[str, Any]
    # Audit log of batches: {start_idx, end_idx, started_at, completed_at, status}.
    batch_log: List[Dict[str, Any]]

    # --- Phase 6: Final output (populated by Publisher) ---
    final_manuscript: Dict[str, Any]

    # --- Metadata (for Django/MongoDB tracking) ---
    status: str  # "pending" | "running" | "awaiting_continue" | "completed" | "failed"
    story_id: str
    error: str
