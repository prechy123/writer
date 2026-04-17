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


# ---------------------------------------------------------------------------
# Continuity ledger (Agent: continuity_extractor) — structured output
# ---------------------------------------------------------------------------

class CharacterState(BaseModel):
    """Snapshot of one character's current state in the story."""

    name: str
    current_location: str = Field(
        default="",
        description="Where this character physically is at the end of the last chapter.",
    )
    emotional_state: str = Field(
        default="",
        description="Current dominant emotion / mental state.",
    )
    knowledge: List[str] = Field(
        default_factory=list,
        description="Plot-relevant facts this character now knows.",
    )
    status: str = Field(
        default="alive",
        description="alive, dead, injured, missing, etc.",
    )
    key_facts: List[str] = Field(
        default_factory=list,
        description="Defining traits or facts introduced about the character so far.",
    )
    last_seen_chapter: int = Field(
        default=0,
        description="Chapter number where the character last appeared.",
    )


class ContinuityLedger(BaseModel):
    """Living continuity document, refreshed after each accepted chapter.

    Enforced via ``llm.with_structured_output(ContinuityLedger)``.
    """

    characters: List[CharacterState] = Field(default_factory=list)
    world_state: str = Field(
        default="",
        description="Time-of-day, current setting, timeline position.",
    )
    items_in_play: List[str] = Field(
        default_factory=list,
        description="MacGuffins, letters, weapons and other objects currently in use.",
    )
    open_threads: List[str] = Field(
        default_factory=list,
        description="Unresolved plot threads. Prefix each with the chapter it was introduced in, e.g. '[Ch.2] Who stole the letter?'",
    )
    resolved_threads: List[str] = Field(
        default_factory=list,
        description="Plot threads that have been closed, with the chapter they were resolved in.",
    )
    foreshadowing_planted: List[str] = Field(
        default_factory=list,
        description="Foreshadowing hooks planted so far, with the chapter they appeared in.",
    )
    named_entities: List[str] = Field(
        default_factory=list,
        description="Proper nouns (people, places, factions, objects) with short definitions, for voice and spelling consistency.",
    )


class ChapterMetadata(BaseModel):
    """Per-chapter record stored alongside the chapter text."""

    chapter_number: int
    title: str
    word_count: int
    summary: str = Field(
        description="2-3 sentence plot summary generated by accept_chapter_node."
    )
    characters_appeared: List[str] = Field(default_factory=list)
    key_events_delivered: List[str] = Field(
        default_factory=list,
        description="Events actually delivered in the prose (may differ from the plan).",
    )
    opening_excerpt: str = Field(
        default="",
        description="First ~300 chars of the chapter (for tone spot-checks).",
    )
    closing_excerpt: str = Field(
        default="",
        description="Last ~800 chars of the chapter (used for stylistic continuity).",
    )
