from typing import List, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Storyteller (Agent 4) — structured output
# ---------------------------------------------------------------------------

class ChapterPlan(BaseModel):
    """Schema for a single chapter inside the story plan."""

    chapter_number: int
    title: str
    summary: str = Field(
        description="2-3 sentence plot summary for this chapter")
    key_events: List[str] = Field(description="Major plot events in order")
    characters_involved: List[str]
    emotional_arc: str = Field(
        description="The emotional trajectory of this chapter "
        "(e.g. 'hopeful → tense → bittersweet')"
    )


class CharacterProfile(BaseModel):
    """Schema for a character in the story bible."""

    name: str
    role: str = Field(description="protagonist, antagonist, supporting, etc.")
    description: str
    motivations: List[str]
    arc: str = Field(description="How this character changes across the story")


class StoryPlanSchema(BaseModel):
    """Full structured output for the Storyteller agent (Agent 4).

    Enforced via ``llm.with_structured_output(StoryPlanSchema)``.
    """

    title: str
    genre: str
    setting: str
    themes: List[str]
    characters: List[CharacterProfile]
    plot_summary: str = Field(
        description="High-level 1-paragraph summary of the entire story"
    )
    chapters: List[ChapterPlan]
    initial_summary: str = Field(
        description="A brief synopsis to initialise the running summary "
        "used by the Writer for context management"
    )


# ---------------------------------------------------------------------------
# Reviewer (Agent 7) — structured output
# ---------------------------------------------------------------------------

class ReviewVerdict(BaseModel):
    """Structured output for the Reviewer agent (Agent 7).

    Enforced via ``llm.with_structured_output(ReviewVerdict)``.
    """

    status: Literal["approve", "revise"] = Field(
        description="'approve' if the chapter is ready, 'revise' if it needs rework"
    )
    feedback: str = Field(description="Detailed feedback for the writer")
    mistakes_found: List[str] = Field(
        default_factory=list,
        description="Specific issues found in the draft",
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="What the draft does well",
    )
