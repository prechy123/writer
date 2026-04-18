from typing import List, Literal, Optional

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
    current_age: str = Field(
        default="",
        description="Character's age at the current point in the timeline. May be "
        "a number ('32') or a range ('mid-40s'). Update if the story time-jumps.",
    )
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
    speech_patterns: List[str] = Field(
        default_factory=list,
        description="Signature proverbs, catchphrases, verbal tics, dialect markers, "
        "or recurring figures of speech this character uses. Keep entries short "
        "(the proverb itself or a brief rule like 'never swears').",
    )
    last_seen_chapter: int = Field(
        default=0,
        description="Chapter number where the character last appeared.",
    )


class CharacterRelationship(BaseModel):
    """The current standing between two characters — the 'Tension' variable."""

    between: List[str] = Field(
        description="The two character names this relationship is between. "
        "Always exactly two names.",
    )
    tension: int = Field(
        ge=1,
        le=10,
        description="Current tension level from 1 (deep trust/love) to 10 "
        "(open hostility/war). 5 is neutral or strangers.",
    )
    description: str = Field(
        description="One-sentence summary of the current dynamic and any "
        "unspoken context (debts, secrets, recent slights).",
    )
    last_updated_chapter: int = Field(
        default=0,
        description="Chapter in which this standing last shifted.",
    )


class SignificantItem(BaseModel):
    """A physical object that matters to the plot — who holds it, where it is, why it matters."""

    name: str
    description: str = Field(
        default="",
        description="What the object is and any physical details.",
    )
    current_holder: str = Field(
        default="",
        description="Character currently in possession, or 'none' if unowned.",
    )
    current_location: str = Field(
        default="",
        description="Where the object physically is right now.",
    )
    significance: str = Field(
        default="",
        description="Why this item matters to the plot (evidence, MacGuffin, heirloom, threat).",
    )


class PlotSeed(BaseModel):
    """A planted mystery, clue, or hint and whether it has paid off yet."""

    description: str = Field(
        description="The seed itself — what was planted, in concrete terms.",
    )
    planted_chapter: int = Field(
        description="Chapter where this seed was planted.",
    )
    payoff_chapter: Optional[int] = Field(
        default=None,
        description="Chapter where this seed was paid off. Null if still dangling.",
    )
    status: Literal["planted", "paid_off", "abandoned"] = Field(
        default="planted",
        description="'planted' = still open; 'paid_off' = resolved; 'abandoned' "
        "= deliberately left unresolved. Never drop a seed — mark it abandoned instead.",
    )


class Subplot(BaseModel):
    """A minor plotline running alongside the main arc."""

    name: str
    status: Literal["active", "dormant", "resolved"] = Field(
        default="active",
        description="'active' = advanced in recent chapters; 'dormant' = not "
        "touched for several chapters but not resolved; 'resolved' = concluded.",
    )
    last_advanced_chapter: int = Field(
        default=0,
        description="Most recent chapter that moved this subplot forward.",
    )
    summary: str = Field(
        default="",
        description="One-sentence state of this subplot right now.",
    )


class WorldRule(BaseModel):
    """A setting-specific fact, law, custom, or system the story must honour."""

    scope: str = Field(
        description="The location, region, faction, or institution this rule applies to "
        "(e.g. 'Umuike village', 'Lagos police checkpoints', 'Enugu State judiciary'). "
        "Use 'global' only when the rule truly applies everywhere in the story.",
    )
    rule: str = Field(
        description="The rule itself — a law, custom, power dynamic, corrupt system, "
        "geography detail, or any worldbuilding fact the writer must not contradict.",
    )
    introduced_in_chapter: int = Field(
        default=0,
        description="Chapter number where this rule was first established.",
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
    world_rules: List[WorldRule] = Field(
        default_factory=list,
        description="Setting-scoped worldbuilding rules (local laws, customs, corrupt "
        "systems, geography). Keep each entry short and tag it to the scope it applies "
        "to so rules from one location never leak into another.",
    )
    significant_items: List[SignificantItem] = Field(
        default_factory=list,
        description="Physical objects that matter to the plot — MacGuffins, letters, "
        "weapons, heirlooms. Track each with its current holder and location.",
    )
    character_relationships: List[CharacterRelationship] = Field(
        default_factory=list,
        description="Pairwise tension/standing between major characters. Only track "
        "relationships that actually matter to the plot — do not enumerate every "
        "possible pair.",
    )
    plot_seeds: List[PlotSeed] = Field(
        default_factory=list,
        description="Mysteries, clues, or hints planted so far. Tracks plant/payoff "
        "so nothing gets introduced and silently forgotten.",
    )
    subplots: List[Subplot] = Field(
        default_factory=list,
        description="Minor plotlines and their current status. Used to rotate "
        "which subplot gets advanced in each chapter.",
    )
    last_chapter_intensity: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Intensity of the most recent chapter on a 1-10 scale "
        "(1 = quiet breather/dialogue-heavy, 10 = climactic/high conflict). "
        "Used to pace the NEXT chapter — avoid two climaxes back-to-back. "
        "0 means no chapter has been written yet.",
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
