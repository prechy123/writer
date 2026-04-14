from typing import Any, Dict, List, TypedDict


class StoryState(TypedDict):
    # --- User inputs ---
    book_title: str
    book_description: str
    num_chapters: int

    # --- Phase 1: Context (populated by Profiler, Empath, Masterclass) ---
    author_profile: str
    emotional_guidelines: str
    expert_styles: str

    # --- Phase 2: Story Architecture (populated by Storyteller) ---
    story_plan: Dict[str, Any]
    chapters_to_write: List[Dict]

    # --- Phase 3-5: Drafting & Review loop ---
    current_chapter_index: int
    current_draft: str
    review_feedback: str
    retry_count: int
    running_summary: str

    # --- Chapter accumulator ---
    final_chapters: List[str]

    # --- Phase 6: Final output (populated by Publisher) ---
    final_manuscript: Dict[str, Any]

    # --- Metadata (for Django/MongoDB tracking) ---
    status: str  # "pending" | "running" | "completed" | "failed"
    story_id: str
    error: str
