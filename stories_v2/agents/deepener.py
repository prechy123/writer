"""Deepener agent — adds interior life to surface-clean prose.

The Editor's job is to FIX specific critic findings. Its prompt explicitly
forbids lengthening the scene, and runs at a temperature aimed at safe
revision. That's appropriate for surface defects (em-dashes, named
emotions, bad pacing) but it leaves a different problem unaddressed:

  - prose that passes every critic
  - feels atmospheric and competent
  - has no interior life — no thought, no contradiction, no memory tied
    to the present moment, no body sensation tied to character history

This pass identifies 1-2 atmospheric passages with no interiority and
injects 1-3 sentences of subjective experience. High temperature on
purpose: the goal is to bring something that wasn't on the page yet,
not to polish what's there. Scope is strict: never rewrite dialogue,
never change events, never grow the scene by more than ~25%.

It runs AFTER the editor + critic loop, BEFORE deterministic humanise.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..providers import Router, get_router
from ..schemas_v2 import CharacterBibleV2, SceneBeat

logger = logging.getLogger(__name__)


SYSTEM = """You are the Deepener. A scene draft is given that is surface-clean but emotionally flat. Your job is to add interior life without rewriting the scene.

WHAT INTERIOR LIFE LOOKS LIKE:
- A specific memory the present moment triggers (not abstract, not flashback — one image, half a beat)
- A body sensation tied to that character's history (the old shoulder injury aches; the hand he writes with goes cold)
- A small contradictory thought (he agrees, then privately thinks the opposite)
- A thing the POV character notices that says more about them than about the room
- A silence between two characters that names something neither said
- A want that surfaces in the wrong moment

WHAT IT IS NOT:
- Naming an emotion ("she felt afraid")
- Authorial mood-painting ("the air was heavy with dread")
- Generic interiority ("he thought about his life")
- Anything that would be true of any character

RULES:
- Add at most 4 short sentences across the whole scene, in 1-2 different places.
- Never change the dialogue.
- Never change what happens. The scene's goal, conflict, and disaster stay identical.
- Never grow the scene by more than ~25% in word count.
- Match the POV character. The added thoughts must sound like THEM, not a generic narrator.
- All the prelude's hard rules still apply: no em-dashes, no banned phrases, no AI-tells, contractions in dialogue (but you're not touching dialogue here).

OUTPUT: the revised prose, start to finish. No commentary. No diff markers. No headers."""


def _compact_beat_for_deepener(beat: SceneBeat) -> Dict[str, Any]:
    keep = (
        "pov_character_name", "kisho_phase", "location", "time_of_day",
        "goal", "conflict", "disaster",
        "protagonist_start_emotion", "protagonist_end_emotion",
        "interiority_density",
    )
    return {k: getattr(beat, k, None) for k in keep if getattr(beat, k, None)}


def _character_voice_anchor(c: CharacterBibleV2) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": c.name, "tier": c.tier.value}
    if c.short_description:
        out["short_description"] = c.short_description
    if c.motivations:
        out["motivations"] = list(c.motivations)[:3]
    if c.fears:
        out["fears"] = list(c.fears)[:2]
    if c.background:
        out["background"] = c.background[:400]
    return out


def _is_atmospheric_only(prose: str) -> bool:
    """Cheap heuristic — does the prose look like all atmosphere, no thought?

    Triggers when there's very little first-person/free-indirect interiority
    or when the prose is dominated by sensory description sentences.
    """
    if not prose or len(prose.split()) < 200:
        return False
    interiority_markers = (
        " he thought",
        " she thought",
        " they thought",
        " he remembered",
        " she remembered",
        " he wanted",
        " she wanted",
        " he hated",
        " she hated",
        " he hoped",
        " she hoped",
        " he wondered",
        " she wondered",
        " he knew",
        " she knew",
        " not that",
        " not now",
        " not yet",
    )
    lower = prose.lower()
    hits = sum(1 for m in interiority_markers if m in lower)
    words = len(prose.split())
    # Less than 1 interiority signal per 500 words → likely atmospheric.
    return hits < max(1, words // 500)


async def deepen_scene(
    *,
    prose: str,
    scene_beat: SceneBeat,
    pov_character: Optional[CharacterBibleV2],
    present_characters: List[CharacterBibleV2],
    overall_critic_score: float,
    has_critic_errors: bool,
    router: Optional[Router] = None,
) -> tuple[str, Dict[str, Any]]:
    """Maybe run a deepening pass. Returns ``(prose, report_dict)``.

    Skips the pass when:
      - the editor still has live errors to fix (let them sort first)
      - the heuristic says the prose has enough interiority already
      - prose is too short to safely add to
    """
    report: Dict[str, Any] = {"ran": False, "reason": ""}
    if has_critic_errors:
        report["reason"] = "skipped: critic errors present (editor will handle)"
        return prose, report
    if not prose or len(prose.split()) < 200:
        report["reason"] = "skipped: prose too short"
        return prose, report
    if not _is_atmospheric_only(prose):
        report["reason"] = "skipped: prose has enough interiority"
        return prose, report

    router = router or get_router()

    pov_anchor = _character_voice_anchor(pov_character) if pov_character else None
    other_anchors = [
        _character_voice_anchor(c)
        for c in present_characters
        if not pov_character or c.character_id != pov_character.character_id
    ][:3]

    import json as _json

    payload_parts: List[str] = []
    payload_parts.append("=== SCENE BEAT ===\n" + _json.dumps(_compact_beat_for_deepener(scene_beat), indent=2, default=str))
    if pov_anchor:
        payload_parts.append("=== POV CHARACTER (anchor the new thoughts here) ===\n" + _json.dumps(pov_anchor, indent=2, default=str))
    if other_anchors:
        payload_parts.append("=== OTHER PRESENT CHARACTERS ===\n" + _json.dumps(other_anchors, indent=2, default=str))
    payload_parts.append("=== CURRENT DRAFT ===\n" + prose)
    payload_parts.append(
        "=== TASK ===\n"
        "The draft is competent on the surface but emotionally flat. Add 1-2 small pockets of "
        "interior life as specified by the system prompt. Touch nothing else. Output the full revised prose."
    )

    try:
        revised = await router.chat_text(
            role="editor",  # reuse the editor role budget
            system=SYSTEM,
            messages=[{"role": "user", "content": "\n\n".join(payload_parts)}],
            max_tokens=_max_tokens_for(scene_beat.target_words),
            temperature=0.95,  # high — we want something that wasn't there before
            timeout=180.0,
        )
    except Exception as exc:
        logger.warning("deepener: LLM call failed (%s); keeping prior draft", exc)
        report.update({"ran": False, "reason": f"llm_failed:{type(exc).__name__}"})
        return prose, report

    revised = (revised or "").strip()
    if not revised:
        report.update({"ran": False, "reason": "empty response"})
        return prose, report

    # Sanity check — if the deepener grew the prose past 25% of original
    # or shrank it past 90%, treat as a failure and keep the original.
    orig_words = max(1, len(prose.split()))
    new_words = len(revised.split())
    growth_ratio = new_words / orig_words
    if growth_ratio > 1.30:
        report.update({"ran": False, "reason": f"rejected: grew {growth_ratio:.2f}x"})
        return prose, report
    if growth_ratio < 0.85:
        report.update({"ran": False, "reason": f"rejected: shrank to {growth_ratio:.2f}x"})
        return prose, report

    report.update({
        "ran": True,
        "reason": "deepened",
        "growth_ratio": round(growth_ratio, 3),
        "orig_word_count": orig_words,
        "new_word_count": new_words,
    })
    return revised, report


def _max_tokens_for(target_words: int) -> int:
    target_words = max(target_words or 600, 300)
    return min(8000, int(target_words * 1.6 * 1.3))
