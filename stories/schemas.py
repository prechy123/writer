from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Storyteller (Agent 4) — structured output
# ---------------------------------------------------------------------------

class WebnovelStrategy(BaseModel):
    """Platform-facing strategy for a Webnovel-style serial."""

    platform_genre: str = Field(
        default="",
        description="Best-fit Webnovel genre/category for discovery.",
    )
    lead_type: str = Field(
        default="",
        description="Reader-facing lead lane, such as male lead, female lead, or ensemble.",
    )
    primary_tags: List[str] = Field(
        default_factory=list,
        description="High-signal discovery tags, e.g. SYSTEM, REVENGE, WEAKTOSTRONG.",
    )
    secondary_tags: List[str] = Field(
        default_factory=list,
        description="Supporting tags that describe tone, relationship dynamics, or tropes.",
    )
    content_rating: str = Field(
        default="",
        description="Reader suitability label such as general, teen, mature, or R18.",
    )
    content_warnings: List[str] = Field(
        default_factory=list,
        description="Concise warnings for violence, abuse, explicit content, or other triggers.",
    )
    target_reader: str = Field(
        default="",
        description="Who the serial is built for and what they already enjoy.",
    )
    reader_promise: str = Field(
        default="",
        description="The repeatable fantasy/emotional reward readers can expect every arc.",
    )
    protagonist_cheat: str = Field(
        default="",
        description="The protagonist's special advantage: system, rebirth knowledge, bloodline, status, skill, etc.",
    )
    self_insertion_anchor: str = Field(
        default="",
        description="Why readers can easily project themselves into the protagonist's position.",
    )
    status_ladder: List[str] = Field(
        default_factory=list,
        description="Visible steps by which the protagonist gains power, money, respect, safety, love, or influence.",
    )
    comparable_hooks: List[str] = Field(
        default_factory=list,
        description="Market-facing hooks comparable to common Webnovel tropes, without copying any protected work.",
    )
    discovery_positioning: str = Field(
        default="",
        description="How the title/synopsis/tags should position the book in browse and ranking contexts.",
    )
    short_logline: str = Field(
        default="",
        description="One-sentence pitch for cards, search results, or social sharing.",
    )
    webnovel_synopsis: str = Field(
        default="",
        description="A Webnovel-style synopsis with immediate conflict, trope clarity, and a strong final hook.",
    )
    title_variants: List[str] = Field(
        default_factory=list,
        description="Alternative clickable Webnovel-style titles.",
    )
    cover_prompt: str = Field(
        default="",
        description="Prompt for generating a cover image that clearly signals genre, lead, and central fantasy.",
    )
    update_schedule: str = Field(
        default="",
        description="Recommended release cadence, e.g. daily, 2 chapters/day, or weekdays.",
    )
    stockpile_strategy: str = Field(
        default="",
        description="How many chapters to prepare before launch and where to place early cliffhangers.",
    )
    ranking_strategy: List[str] = Field(
        default_factory=list,
        description="Actions that help collection, power, update, active, and comment rankings.",
    )
    author_note_templates: List[str] = Field(
        default_factory=list,
        description="Short reusable author notes for comments, library adds, and votes.",
    )


class LaunchChapterPlan(BaseModel):
    """Reader-conversion plan for Chapter 1 and the first upload batch."""

    conversion_goal: str = Field(
        default="",
        description="What Chapter 1 must make a browsing reader feel or decide before leaving.",
    )
    first_line_strategy: str = Field(
        default="",
        description="How the opening line should create instant pressure, desire, image, or question.",
    )
    first_200_words_hook: str = Field(
        default="",
        description="The exact hook Chapter 1 must establish in the first 200 words.",
    )
    first_scene_pressure: str = Field(
        default="",
        description="Immediate conflict, humiliation, danger, desire, betrayal, countdown, or social pressure in scene one.",
    )
    protagonist_snapshot: str = Field(
        default="",
        description="What readers should understand about the protagonist's wound, desire, limitation, and appeal.",
    )
    reader_sympathy_trigger: str = Field(
        default="",
        description="The moment that makes readers emotionally side with the protagonist.",
    )
    special_edge_tease: str = Field(
        default="",
        description="How Chapter 1 should hint at the protagonist's system, rebirth knowledge, rare talent, hidden status, or other edge.",
    )
    stakes_lock: str = Field(
        default="",
        description="The concrete consequence that makes the reader understand why the story must continue now.",
    )
    inciting_turn: str = Field(
        default="",
        description="The turn that shifts Chapter 1 from setup into an unavoidable serial engine.",
    )
    chapter_one_progression_reward: str = Field(
        default="",
        description="The visible win, reveal, clue, status shift, or leverage Chapter 1 must deliver.",
    )
    chapter_one_cliffhanger: str = Field(
        default="",
        description="The Chapter 1 ending beat that should push readers into Chapter 2.",
    )
    first_five_chapter_promises: List[str] = Field(
        default_factory=list,
        description="Reader-facing promises the first five chapters should make and begin satisfying.",
    )
    tag_delivery_moments: List[str] = Field(
        default_factory=list,
        description="Concrete early scenes that prove the chosen Webnovel tags are real.",
    )
    comment_magnet_question: str = Field(
        default="",
        description="A genuine post-chapter question likely to invite reader comments without begging.",
    )
    early_dropoff_risks: List[str] = Field(
        default_factory=list,
        description="Specific launch weaknesses to avoid: slow setup, unclear cheat, passive lead, genre mismatch, etc.",
    )
    revision_checklist: List[str] = Field(
        default_factory=list,
        description="Chapter-1 review checklist used by Writer, Reviewer, and Perfectionist.",
    )


class SerialArc(BaseModel):
    """Long-form arc roadmap for a serial that may run beyond the first batch."""

    arc_number: int
    title: str
    chapter_range: str = Field(
        description="Human-readable chapter range, e.g. '1-25'.",
    )
    external_goal: str = Field(
        description="Concrete goal the protagonist is chasing in this arc.",
    )
    central_conflict: str = Field(
        description="Main opposition, dilemma, or relationship pressure of the arc.",
    )
    progression_rewards: List[str] = Field(
        default_factory=list,
        description="Power-ups, reveals, status gains, relationship shifts, or wins promised in the arc.",
    )
    major_reversals: List[str] = Field(
        default_factory=list,
        description="Setbacks or twists that prevent the arc from becoming linear.",
    )
    cliffhanger_bridge: str = Field(
        default="",
        description="How this arc pulls the reader into the next arc.",
    )


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
    opening_hook: str = Field(
        default="",
        description="The first-scene hook or immediate question that should grab readers.",
    )
    progression_reward: str = Field(
        default="",
        description="Concrete reward delivered in this chapter: new power, clue, win, status, intimacy, or leverage.",
    )
    new_question_raised: str = Field(
        default="",
        description="Fresh mystery, danger, desire, or uncertainty created before the chapter ends.",
    )
    cliffhanger: str = Field(
        default="",
        description="The final beat that should make readers open the next chapter.",
    )
    reader_emotion_target: str = Field(
        default="",
        description="Dominant reader feeling to create: outrage, triumph, dread, yearning, curiosity, etc.",
    )
    tags_served: List[str] = Field(
        default_factory=list,
        description="Which Webnovel discovery tags/tropes this chapter actively satisfies.",
    )
    comment_prompt: str = Field(
        default="",
        description="Short optional author-note prompt to invite comments after this chapter.",
    )
    power_stone_pitch: str = Field(
        default="",
        description="Short optional author-note prompt to invite votes/support after this chapter.",
    )
    filler_risk: str = Field(
        default="",
        description="Specific thing this chapter must avoid so it does not feel like filler.",
    )


class CharacterProfile(BaseModel):
    """Schema for a character in the story bible."""

    name: str
    role: str = Field(description="protagonist, antagonist, supporting, etc.")
    description: str
    motivations: List[str]
    arc: str = Field(description="How this character changes across the story")
    webnovel_role_hook: str = Field(
        default="",
        description="Reader-facing trope/function this character serves: rival, hidden master, jealous sibling, cold CEO, etc.",
    )
    progression_function: str = Field(
        default="",
        description="How this character creates wins, setbacks, training, status shifts, romance tension, or reveals.",
    )
    speech_style: str = Field(
        default="",
        description="How this character sounds in dialogue: diction, rhythm, "
        "formality, slang, restraint, favourite rhetorical habits, and what "
        "they would never say.",
    )
    education_access: str = Field(
        default="",
        description="What education or informal learning this character has "
        "realistically accessed, including gaps or limits.",
    )
    resources_or_limitations: str = Field(
        default="",
        description="Money, technology, family support, social access, mobility, "
        "or other constraints that should shape what this character can plausibly do.",
    )
    knowledge_sources: List[str] = Field(
        default_factory=list,
        description="Concrete sources for what this character knows: school, "
        "libraries, borrowed books, radio, TV viewing centers, mentors, work, "
        "internet cafes, phones, apprenticeships, or similar.",
    )


class StoryPlanSchema(BaseModel):
    """Full structured output for the Storyteller agent (Agent 4).

    Enforced via ``llm.with_structured_output(StoryPlanSchema)``.
    """

    title: str
    genre: str
    webnovel_strategy: WebnovelStrategy = Field(
        default_factory=WebnovelStrategy,
        description="Discovery, retention, and publishing strategy for Webnovel.",
    )
    launch_chapter_plan: LaunchChapterPlan = Field(
        default_factory=LaunchChapterPlan,
        description="Dedicated conversion plan for Chapter 1 and the first upload batch.",
    )
    setting: str
    themes: List[str]
    characters: List[CharacterProfile]
    plot_summary: str = Field(
        description="High-level 1-paragraph summary of the entire story"
    )
    chapters: List[ChapterPlan]
    serial_arcs: List[SerialArc] = Field(
        default_factory=list,
        description="Arc roadmap. First arc should align with the detailed chapter list; later arcs may be broader.",
    )
    release_plan: List[str] = Field(
        default_factory=list,
        description="Launch and update plan: stockpile, daily cadence, early hook milestones, batch strategy.",
    )
    retention_strategy: List[str] = Field(
        default_factory=list,
        description="Rules for keeping readers through comments, cliffhangers, progression, and emotional payoffs.",
    )
    long_form_roadmap: str = Field(
        default="",
        description="How this premise can expand beyond the requested chapters without losing focus.",
    )
    opening_strategy_notes: List[str] = Field(
        default_factory=list,
        description="Book-level guidance for varying chapter openings. Avoid "
        "reusing the same first-image, sentence rhythm, punch line, or emotional beat.",
    )
    background_constraints: List[str] = Field(
        default_factory=list,
        description="Continuity constraints from the user's background and the "
        "story premise, especially socioeconomic access, education, technology, "
        "travel, housing, and media exposure limits.",
    )
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
    knowledge_sources: List[str] = Field(
        default_factory=list,
        description="Concrete sources that explain how this character learned "
        "important facts or gained access to books, media, technology, mentors, "
        "schooling, work experience, or specialist information.",
    )
    resource_constraints: List[str] = Field(
        default_factory=list,
        description="Current money, technology, mobility, family, class, housing, "
        "or social-access limits the story must not contradict.",
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
    reader_open_loops: List[str] = Field(
        default_factory=list,
        description="Current reader-facing questions, cliffhangers, threats, promises, or desires pulling readers forward.",
    )
    progression_milestones: List[str] = Field(
        default_factory=list,
        description="Visible wins and growth steps already delivered: power, status, money, romance, skill, clue, territory, safety.",
    )
    unresolved_cliffhangers: List[str] = Field(
        default_factory=list,
        description="Specific cliffhangers introduced at chapter endings that still need payoff.",
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
    launch_plan_applied: bool = Field(
        default=False,
        description="True when the dedicated Chapter 1 launch plan was active for this chapter.",
    )
    first_200_word_goal: str = Field(
        default="",
        description="Launch-plan target for the first 200 words, when applicable.",
    )
    comment_magnet_question: str = Field(
        default="",
        description="Launch-plan comment prompt associated with this chapter, when applicable.",
    )
    opening_excerpt: str = Field(
        default="",
        description="First ~300 chars of the chapter (for tone spot-checks).",
    )
    closing_excerpt: str = Field(
        default="",
        description="Last ~800 chars of the chapter (used for stylistic continuity).",
    )
    summary_embedding: List[float] = Field(
        default_factory=list,
        description="Vector embedding of ``summary``. Populated by accept_chapter_node "
        "and used by stories/retrieval.py to pick the most relevant earlier "
        "chapters for the upcoming Writer prompt. Empty when embedding is "
        "unavailable; the retrieval layer treats that as 'skip this entry'.",
    )
