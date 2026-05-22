"""Scene Writer prompt — the one that produces prose.

This is the single most important prompt in the pipeline. Built from:
  - system_prelude (humanisation rules)
  - role-specific writer instructions
  - voice few-shot anchors for present characters
  - corpus exemplar passages (technique anchors)
  - memory context (working tier verbatim)
  - scene beat (G-C-D + emotion target + sensory focus)

We deliberately keep instruction text short. The bulk of the prompt is
demonstration: few-shot character voice + corpus exemplars. Models
mimic what they see better than what they're told.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from .system_prelude import build_system

ROLE = """You are the Scene Writer. Write ONE scene of a web novel.

You are not narrating. You are not summarising. You are placing the reader inside the scene in real time. Drop them in mid-sense, mid-thought, or mid-action. They catch up.

THIS SCENE'S CONTRACT:
- Hit the scene beat (Goal / Conflict / Disaster). The scene ENDS worse, complicated, or escalated. Never neatly resolved.
- Move the POV character's emotional state from the planned start toward the planned end. Show the delta in body, breath, action, and unspoken thought. Not in stated feelings.
- Anchor at least one of the listed sensory_focus axes hard. Make the reader smell / hear / feel something concrete and specific.
- Match the planned interiority_density: low = action-led, medium = balanced, high = inside the character's head.
- Apply the listed techniques. The corpus exemplars below illustrate them.
- Match each speaking character's voice fingerprint. Use the few-shot lines as anchors for tone, contractions, register, length.

OUTPUT FORMAT: prose only. Just the scene. No headers. No commentary. No "Scene 3:" or "Chapter 4". No editor notes. No bullet points. Begin the prose where the scene begins.

LENGTH: aim near the target_words number; the band is soft. A tight scene at 75% of target is better than a bloated scene at 110%. Do NOT pad.

CRITICAL — already in the system prelude but worth repeating here: do not use em-dashes. Do not use the AI-tell phrases. Vary sentence length. Use contractions. Trust the reader.
"""

SYSTEM = build_system(ROLE)


def build_user_prompt(
    *,
    scene_beat: Dict[str, Any],
    voice_few_shot_block: str,
    corpus_exemplars: List[Dict[str, Any]],
    working_memory: Dict[str, Any],
    semantic_context: Dict[str, Any],
    episodic_excerpts: List[Dict[str, Any]],
    author_profile_hint: Optional[Dict[str, Any]] = None,
    continuation_brief: Optional[str] = None,
) -> str:
    parts: List[str] = []

    if continuation_brief and continuation_brief.strip():
        parts.append(
            "=== AUTHOR'S CONTINUATION BRIEF (overall plot intent across upcoming chapters) ===\n"
            + continuation_brief.strip()
        )

    parts.append("=== SCENE BEAT ===\n" + json.dumps(_compact_beat(scene_beat), indent=2, default=str))

    if voice_few_shot_block:
        parts.append("=== CHARACTER VOICES (few-shot anchors) ===\n" + voice_few_shot_block)

    if corpus_exemplars:
        block_lines = []
        for ex in corpus_exemplars:
            techniques = ", ".join(ex.get("techniques") or [])
            block_lines.append(f"-- exemplar {ex.get('id', '')} [{techniques}] --\n{ex.get('text', '').strip()}")
        parts.append("=== CRAFT EXEMPLARS (mimic technique, not content) ===\n" + "\n\n".join(block_lines))

    if working_memory.get("previous_scene_ending"):
        parts.append("=== PREVIOUS SCENE ENDING (carry continuity) ===\n" + working_memory["previous_scene_ending"])

    if working_memory.get("last_chapter_excerpts"):
        chunks = []
        for ch in working_memory["last_chapter_excerpts"]:
            chunks.append(f"-- chapter {ch.get('chapter_idx')} head --\n{ch.get('head', '')[:1500]}")
            if ch.get("tail"):
                chunks.append(f"-- chapter {ch.get('chapter_idx')} tail --\n{ch.get('tail', '')[:1500]}")
        parts.append("=== RECENT CHAPTER EXCERPTS ===\n" + "\n\n".join(chunks))

    if episodic_excerpts:
        chunks = []
        for ex in episodic_excerpts:
            ch = ex.get("chapter_idx", "?")
            sc = ex.get("scene_idx", "?")
            summary = ex.get("summary") or ""
            chunks.append(f"-- ch{ch} sc{sc} summary --\n{summary}")
        parts.append("=== RELEVANT EARLIER SCENES (for continuity) ===\n" + "\n\n".join(chunks))

    sem = semantic_context or {}
    if sem.get("present_characters"):
        chars = sem["present_characters"]
        chunks = []
        for c in chars:
            chunks.append(_compact_character(c))
        parts.append("=== PRESENT CHARACTERS ===\n" + "\n\n".join(chunks))

    if sem.get("world_bible"):
        wb = sem["world_bible"]
        compact = {
            k: wb.get(k)
            for k in ("setting", "magic_or_system", "banned_anachronisms", "must_have_vibes")
            if wb.get(k)
        }
        if compact:
            parts.append("=== WORLD ANCHORS ===\n" + json.dumps(compact, indent=2, default=str))

    if sem.get("relevant_world_rules"):
        parts.append(
            "=== WORLD RULES IN SCOPE ===\n"
            + json.dumps(sem["relevant_world_rules"], indent=2, default=str)
        )

    threads = sem.get("active_threads") or {}
    if threads.get("unresolved_cliffhangers"):
        parts.append("=== UNRESOLVED CLIFFHANGERS (keep alive or pay off) ===\n"
                     + json.dumps(threads["unresolved_cliffhangers"], indent=2, default=str))

    if author_profile_hint:
        parts.append(
            "=== AUTHOR VOICE HINT (apply to narrative voice, not character dialogue) ===\n"
            + json.dumps(_compact_profile(author_profile_hint), indent=2, default=str)
        )

    parts.append(
        "=== WRITE THE SCENE NOW ===\n"
        "Open mid-moment. Hit the beat. End on the disaster. Prose only."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Compactors
# ---------------------------------------------------------------------------

def _compact_beat(beat: Dict[str, Any]) -> Dict[str, Any]:
    keep = (
        "scene_idx", "title", "summary", "kisho_phase",
        "pov_character_id", "pov_character_name", "present_character_ids",
        "location", "time_of_day",
        "goal", "conflict", "disaster",
        "protagonist_start_emotion", "protagonist_end_emotion",
        "reader_start_emotion", "reader_end_emotion",
        "sensory_focus", "interiority_density", "techniques",
        "target_words",
    )
    return {k: beat.get(k) for k in keep if beat.get(k) is not None}


def _compact_character(c: Dict[str, Any]) -> str:
    lines = [f"{c.get('name', '?')} ({c.get('tier', 'recurring')}; {c.get('role', '')})"]
    if c.get("short_description"):
        lines.append(c["short_description"])
    if c.get("motivations"):
        lines.append("Motivations: " + "; ".join(c["motivations"][:3]))
    if c.get("fears"):
        lines.append("Fears: " + "; ".join(c["fears"][:2]))
    vf = c.get("voice_fingerprint") or {}
    lex = vf.get("lexical") or {}
    if lex:
        bits = []
        if lex.get("style_register"):
            bits.append(f"register={lex['style_register']}")
        if lex.get("contraction_rate") is not None:
            bits.append(f"contract={lex['contraction_rate']:.2f}")
        if lex.get("formality") is not None:
            bits.append(f"formality={lex['formality']}/10")
        if bits:
            lines.append("Speech: " + ", ".join(bits))
    return "\n  ".join(lines)


def _compact_profile(p: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("preferred_phrases", "banned_phrases", "lexical_fingerprint", "emotional_defaults")
    return {k: p.get(k) for k in keep if p.get(k)}
