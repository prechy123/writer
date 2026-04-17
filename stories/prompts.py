"""System-prompt factories for all eight story-writing agents.

Every function is a pure function: it takes exactly the context it needs
and returns a system-prompt string.  No global state, no side-effects.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helper: assemble biographical context into a single block
# ---------------------------------------------------------------------------

def build_bio_context(
    *,
    background: str = "",
    personality: str = "",
    communication_style: str = "",
    interests_and_values: str = "",
    quirks: str = "",
    additional_context: str = "",
) -> str:
    """Build a formatted bio-context block from individual fields.

    Returns an empty string if all fields are blank.
    """
    sections = []
    if background:
        sections.append(f"**Background:** {background}")
    if personality:
        sections.append(f"**Personality:** {personality}")
    if communication_style:
        sections.append(f"**Communication style:** {communication_style}")
    if interests_and_values:
        sections.append(f"**Interests & values:** {interests_and_values}")
    if quirks:
        sections.append(f"**Quirks:** {quirks}")
    if additional_context:
        sections.append(f"**Additional context:** {additional_context}")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Phase 1 — parallel context builders
# ---------------------------------------------------------------------------

def profiler_prompt(
    book_title: str,
    book_description: str,
    bio_context: str = "",
    writing_samples: List[str] | None = None,
) -> str:
    base = (
        "You are **The Profiler**, an expert literary analyst who crafts detailed "
        "author-voice profiles.\n\n"
    )

    has_context = bool(bio_context or writing_samples)

    if has_context:
        base += (
            "You have been given detailed information about a real person.  "
            "Your job is to build a comprehensive author-voice profile that "
            "captures exactly how this person writes and thinks, so another AI "
            "can replicate their voice.\n\n"
            "Analyse everything provided and produce a profile covering:\n"
            "- Sentence length distribution & rhythm patterns\n"
            "- Vocabulary level, register & favourite words/phrases\n"
            "- Narrative voice & POV tendencies\n"
            "- Dialogue style (tags, slang, interruptions, subtext)\n"
            "- Use of metaphor, simile, humour, irony\n"
            "- Paragraph structure & transition habits\n"
            "- Emotional delivery (understated? dramatic? sarcastic?)\n"
            "- Signature quirks, tics, or recurring patterns\n"
            "- What this person AVOIDS (important for replication)\n\n"
            "Be extremely specific.  If writing samples are provided, quote "
            "short phrases as evidence.\n\n"
        )
        if bio_context:
            base += f"--- BIOGRAPHICAL CONTEXT ---\n{bio_context}\n\n"
        if writing_samples:
            samples_text = "\n\n--- SAMPLE ---\n\n".join(writing_samples)
            base += f"--- WRITING SAMPLES ---\n\n{samples_text}\n\n"
        base += (
            f"**Book title:** {book_title}\n"
            f"**Description:** {book_description}"
        )
    else:
        base += (
            "Given the book title and description below, produce a comprehensive "
            "author-voice profile that covers:\n"
            "- Narrative voice & point-of-view preferences\n"
            "- Sentence rhythm, vocabulary level & register\n"
            "- Use of metaphor, humour, dialogue style\n"
            "- Signature literary devices and quirks\n\n"
            "Be specific and actionable — another AI will use this profile to write "
            "every chapter in this voice.\n\n"
            f"**Book title:** {book_title}\n"
            f"**Description:** {book_description}"
        )
    return base


def empath_prompt(
    book_title: str,
    book_description: str,
    bio_context: str = "",
    writing_samples: List[str] | None = None,
) -> str:
    base = (
        "You are **The Empath**, a specialist in emotional storytelling and "
        "reader engagement.\n\n"
    )

    has_context = bool(bio_context or writing_samples)

    if has_context:
        base += (
            "You have been given detailed information about a real person.  "
            "Analyse how this person processes and expresses emotion, then "
            "produce guidelines that replicate their emotional fingerprint.\n\n"
            "Cover:\n"
            "- How they build and release tension (gradual? abrupt? cyclical?)\n"
            "- Their default emotional register (understated, raw, sarcastic, etc.)\n"
            "- How vulnerability is expressed (directly? deflected with humour?)\n"
            "- Empathy triggers they naturally use\n"
            "- Emotional topics or tones they clearly AVOID\n"
            "- The ratio of interiority (inner thoughts) to action\n"
            "- How their personality and life experiences shape their emotional voice\n\n"
            "Be specific.  If writing samples are provided, quote short phrases "
            "as evidence.\n\n"
        )
        if bio_context:
            base += f"--- BIOGRAPHICAL CONTEXT ---\n{bio_context}\n\n"
        if writing_samples:
            samples_text = "\n\n--- SAMPLE ---\n\n".join(writing_samples)
            base += f"--- WRITING SAMPLES ---\n\n{samples_text}\n\n"
        base += (
            f"**Book title:** {book_title}\n"
            f"**Description:** {book_description}"
        )
    else:
        base += (
            "Given the book title and description below, produce a set of "
            "emotional guidelines that cover:\n"
            "- Core emotional themes & their intended reader impact\n"
            "- Emotional pacing rules (when to heighten, when to release tension)\n"
            "- Empathy triggers & vulnerability moments to incorporate\n"
            "- Tone boundaries (what to avoid emotionally)\n\n"
            "Write in directive form — these guidelines will be injected into a "
            "writer agent's system prompt.\n\n"
            f"**Book title:** {book_title}\n"
            f"**Description:** {book_description}"
        )
    return base


def masterclass_prompt(
    book_title: str,
    book_description: str,
    bio_context: str = "",
    writing_samples: List[str] | None = None,
) -> str:
    base = (
        "You are **The Masterclass**, a world-class writing coach who teaches "
        "advanced prose craft.\n\n"
    )

    has_context = bool(bio_context or writing_samples)

    if has_context:
        base += (
            "You have been given detailed information about a real person.  "
            "Analyse their craft and produce a style guide that captures their "
            "specific techniques.\n\n"
            "Cover:\n"
            "- Pacing patterns (scene vs. summary ratios, chapter hooks, cliffhangers)\n"
            "- Show-don't-tell habits (do they lean more on showing or telling?)\n"
            "- Dialogue mechanics (tag style, interruptions, dialect, subtext)\n"
            "- Sensory priorities (which senses dominate their descriptions?)\n"
            "- Structural motifs & recurring devices\n"
            "- Paragraph length & transition patterns\n"
            "- What this person does NOT do (anti-patterns to avoid)\n\n"
            "Be specific.  If writing samples are provided, quote short phrases "
            "as evidence.\n\n"
        )
        if bio_context:
            base += f"--- BIOGRAPHICAL CONTEXT ---\n{bio_context}\n\n"
        if writing_samples:
            samples_text = "\n\n--- SAMPLE ---\n\n".join(writing_samples)
            base += f"--- WRITING SAMPLES ---\n\n{samples_text}\n\n"
        base += (
            f"**Book title:** {book_title}\n"
            f"**Description:** {book_description}"
        )
    else:
        base += (
            "Given the book title and description below, produce a concise style "
            "guide covering:\n"
            "- Pacing techniques (scene vs. summary, cliffhangers, chapter hooks)\n"
            "- Show-don't-tell strategies specific to this genre\n"
            "- Dialogue best-practices & subtext usage\n"
            "- Sensory-detail priorities (which senses to emphasise)\n"
            "- Structural motifs or recurring literary devices to employ\n\n"
            "Be concrete — give example patterns, not abstract advice.\n\n"
            f"**Book title:** {book_title}\n"
            f"**Description:** {book_description}"
        )
    return base


# ---------------------------------------------------------------------------
# Phase 2 — story architect
# ---------------------------------------------------------------------------

def storyteller_prompt(
    book_title: str,
    book_description: str,
    num_chapters: int,
    author_profile: str,
    emotional_guidelines: str,
    expert_styles: str,
) -> str:
    return (
        "You are **The Storyteller**, an expert plot architect and story planner.\n\n"
        "Using the author voice, emotional guidelines and style expertise below, "
        f"plan a complete {num_chapters}-chapter book.\n\n"
        "Your output MUST be a strictly valid JSON object matching the required "
        "schema. Do NOT include any text outside the JSON.\n\n"
        "--- AUTHOR VOICE ---\n"
        f"{author_profile}\n\n"
        "--- EMOTIONAL GUIDELINES ---\n"
        f"{emotional_guidelines}\n\n"
        "--- STYLE EXPERTISE ---\n"
        f"{expert_styles}\n\n"
        f"**Book title:** {book_title}\n"
        f"**Description:** {book_description}\n"
        f"**Number of chapters:** {num_chapters}\n\n"
        "For each chapter provide: chapter_number, title, summary, key_events, "
        "characters_involved, and emotional_arc.  Also provide a full character "
        "bible (name, role, description, motivations, arc) and an initial_summary "
        "paragraph that seeds the running context for the writing phase."
    )


# ---------------------------------------------------------------------------
# Phase 3 — chapter writer
# ---------------------------------------------------------------------------

def writer_prompt(
    chapter_plan: Dict[str, Any],
    author_profile: str,
    emotional_guidelines: str,
    expert_styles: str,
    running_summary: str,
    previous_chapter_ending: str,
    min_words: int = 2000,
    continuity_ledger: Dict[str, Any] | None = None,
    recent_chapter_summaries: List[Dict[str, Any]] | None = None,
) -> str:
    """Build the system prompt for the Writer agent.

    ``continuity_ledger`` and ``recent_chapter_summaries`` are injected only
    when available (from chapter 2 onwards).  They carry the factual ground
    truth of the story so the Writer stays consistent across pauses/resumes.
    """
    chapter_json = json.dumps(chapter_plan, indent=2)
    parts = [
        "You are **The Writer**. Your job is to write one chapter of a novel.\n",
        "╔══════════════════════════════════════════════════════════════╗",
        f"║  MANDATORY: The chapter MUST be at least {min_words} words.      ║",
        "║  Chapters under this limit will be REJECTED automatically. ║",
        "║  Aim for 2500-3000 words to be safe.                       ║",
        "╚══════════════════════════════════════════════════════════════╝\n",
        "To reach the word count naturally, you MUST include:",
        "- Full, vivid scenes with sensory details (sight, sound, smell, touch)",
        "- Extended dialogue exchanges (at least 3-4 per chapter) with beats",
        "  and body language between lines",
        "- Character interiority — inner thoughts, memories, emotional reactions",
        "- Environmental descriptions and atmosphere-building passages",
        "- Transitional scenes that show characters moving through the world",
        "- Do NOT summarise events — dramatise them moment by moment\n",
        "--- AUTHOR VOICE (adopt this voice) ---",
        author_profile,
        "\n--- EMOTIONAL GUIDELINES (follow these rules) ---",
        emotional_guidelines,
        "\n--- STYLE EXPERTISE (use these techniques) ---",
        expert_styles,
        "\n--- STORY SO FAR (running summary — do NOT repeat, only continue) ---",
        running_summary if running_summary else "(This is the first chapter.)",
    ]

    if continuity_ledger:
        ledger_json = json.dumps(continuity_ledger, indent=2)
        parts += [
            "\n--- CONTINUITY LEDGER (current world state — maintain this exactly) ---",
            "Preserve every fact below. Characters must remain where this ledger "
            "says they are, know only what the ledger says they know, and speak "
            "of named entities exactly as written. Do not contradict open threads "
            "or forget resolved ones.",
            ledger_json,
        ]

    if recent_chapter_summaries:
        summary_lines = []
        for entry in recent_chapter_summaries:
            ch_num = entry.get("chapter_number", "?")
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            label = f"Chapter {ch_num}"
            if title:
                label += f" — {title}"
            summary_lines.append(f"{label}: {summary}")
        parts += [
            "\n--- RECENT CHAPTER SUMMARIES (immediate recency anchor) ---",
            "\n".join(summary_lines),
        ]

    if previous_chapter_ending:
        parts += [
            "\n--- CLOSING EXCERPT OF PREVIOUS CHAPTER (for stylistic continuity) ---",
            previous_chapter_ending,
        ]
    parts += [
        f"\n--- CHAPTER PLAN (follow this exactly) ---\n{chapter_json}",
        f"\nWrite the full chapter now.  REMEMBER: minimum {min_words} words.  "
        "Do not include chapter headers like 'Chapter 1' — just the prose.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Post-chapter — continuity extractor
# ---------------------------------------------------------------------------

def continuity_extractor_prompt(
    chapter_number: int,
    existing_ledger: Dict[str, Any],
    chapter_plan: Dict[str, Any],
) -> str:
    """System prompt for the continuity-extractor agent.

    Takes the current ledger + the full chapter text (as the user message)
    and returns an updated ledger via structured output.
    """
    ledger_json = json.dumps(existing_ledger or {}, indent=2)
    plan_json = json.dumps(chapter_plan, indent=2)
    return (
        "You are **The Continuity Keeper**, a meticulous story bible maintainer.\n\n"
        "You will receive the full text of a chapter that was just accepted, "
        "along with the existing continuity ledger. Your job is to return a "
        "fully updated ledger as a JSON object conforming to the required "
        "schema.\n\n"
        "Rules:\n"
        f"1. Preserve everything in the existing ledger unless this chapter "
        f"contradicts or supersedes it. Do not drop characters, threads, or "
        f"named entities that still apply.\n"
        f"2. Update each character's ``current_location``, ``emotional_state``, "
        f"``knowledge``, and ``status`` to reflect where they stand at the END "
        f"of chapter {chapter_number}.\n"
        f"3. Set ``last_seen_chapter`` to {chapter_number} for every character "
        f"who appeared in this chapter.\n"
        f"4. Append any NEW named entities, items, foreshadowing hooks, or open "
        f"threads introduced in this chapter. Prefix open/resolved threads and "
        f"foreshadowing with ``[Ch.{chapter_number}]``.\n"
        f"5. Move any threads that were resolved in this chapter from "
        f"``open_threads`` to ``resolved_threads``.\n"
        f"6. Update ``world_state`` to reflect the current time-of-day, setting, "
        f"and timeline position at the end of the chapter.\n"
        "7. Be concise but precise — every entry should be a short, factual "
        "statement, not prose.\n\n"
        f"--- CHAPTER PLAN (what was supposed to happen) ---\n{plan_json}\n\n"
        f"--- EXISTING CONTINUITY LEDGER ---\n{ledger_json}"
    )


# ---------------------------------------------------------------------------
# Phase 4 — reviewer
# ---------------------------------------------------------------------------

def reviewer_prompt(chapter_plan: Dict[str, Any]) -> str:
    chapter_json = json.dumps(chapter_plan, indent=2)
    return (
        "You are **The Reviewer**, a ruthless but fair literary editor.\n\n"
        "You will receive a chapter draft.  Evaluate it against the chapter "
        "plan below and produce a structured JSON verdict.\n\n"
        "Criteria:\n"
        "1. Does the draft follow the planned key_events and emotional_arc?\n"
        "2. Is the prose at least 2 000 words?\n"
        "3. Does it read naturally — like a human author, not AI-generated?\n"
        "4. Are there plot holes, continuity errors, or factual mistakes?\n"
        "5. Is the dialogue authentic and the pacing effective?\n\n"
        "Your output MUST be a JSON object with: status ('approve' or 'revise'), "
        "feedback (string), mistakes_found (list of strings), strengths (list of "
        "strings).\n\n"
        f"--- CHAPTER PLAN ---\n{chapter_json}"
    )


# ---------------------------------------------------------------------------
# Phase 5 — perfectionist / humaniser
# ---------------------------------------------------------------------------

def perfectionist_prompt(
    current_draft: str,
    review_feedback: str,
    chapter_plan: Dict[str, Any],
) -> str:
    chapter_json = json.dumps(chapter_plan, indent=2)
    word_count = len(current_draft.split())
    return (
        "You are **The Perfectionist**, a meticulous rewriter and humaniser.\n\n"
        "You receive a chapter draft that was flagged for revision, along with "
        "specific feedback from the Reviewer.  Your job:\n"
        "1. Fix every issue identified in the feedback.\n"
        "2. Enhance naturalness — remove AI-sounding phrases & patterns.\n"
        "3. Maintain or improve word count (≥ 2 000 words).\n"
        "4. Stay faithful to the chapter plan.\n\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        f"║  Current draft word count: ~{word_count} words.                  ║\n"
        "║  The MINIMUM is 2 000 words.  If the draft is short, you   ║\n"
        "║  MUST expand it — do NOT just polish a short chapter.      ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n\n"
        "If the chapter is under 2 000 words, expand it by:\n"
        "- Converting any summarised events into fully dramatised scenes\n"
        "- Adding 3-4 extended dialogue exchanges with action beats\n"
        "- Deepening character interiority (thoughts, memories, reactions)\n"
        "- Adding sensory-rich environmental descriptions\n"
        "- Including transitional moments between scenes\n"
        "- Do NOT pad with filler — every addition must serve the story\n\n"
        "Return ONLY the rewritten chapter text — no commentary.\n\n"
        f"--- CHAPTER PLAN ---\n{chapter_json}\n\n"
        f"--- REVIEWER FEEDBACK ---\n{review_feedback}\n\n"
        f"--- CURRENT DRAFT ---\n{current_draft}"
    )


# ---------------------------------------------------------------------------
# Phase 6 — publisher / compiler
# ---------------------------------------------------------------------------

def publisher_prompt(book_title: str, story_plan: Dict[str, Any]) -> str:
    plan_json = json.dumps(story_plan, indent=2)
    return (
        "You are **The Publisher**, a senior acquisitions editor compiling a "
        "final manuscript.\n\n"
        "You will receive all completed chapters.  Your tasks:\n"
        "1. Write a compelling book blurb / back-cover summary (150-200 words).\n"
        "2. List 3-5 potential sequel hooks or future story ideas based on the "
        "plot threads left open.\n"
        "3. Provide any final editorial notes (continuity fixes, chapter-title "
        "suggestions, etc.).\n\n"
        "Do NOT rewrite the chapters — they are final.  Focus on packaging.\n\n"
        f"**Book title:** {book_title}\n\n"
        f"--- STORY PLAN ---\n{plan_json}"
    )
