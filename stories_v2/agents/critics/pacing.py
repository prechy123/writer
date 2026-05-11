"""Pacing critic — opening hook, GCD presence, length feel, cliffhanger pull."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ...prompts_v2.critics import PACING_CRITIC_SYSTEM, build_critic_user_prompt
from ...providers import Router
from ...schemas_v2 import CriticFinding, CriticReport
from ._common import call_critic_llm, compact_scene_beat, merge_reports, parse_critic_report

_WORD = re.compile(r"\b[\w'-]+\b")


def _heuristic_pacing(draft: str, scene_beat: Dict[str, Any]) -> CriticReport:
    """Structural checks that don't need the LLM.

    The qualitative judgement (rushed vs. tight vs. padded) is the LLM's
    job; here we only flag the clear structural failures.
    """
    findings: List[CriticFinding] = []
    text = (draft or "").strip()
    if not text:
        return CriticReport(
            critic="pacing",
            score=0.0,
            on_target=False,
            findings=[CriticFinding(
                critic="pacing",
                severity="error",
                field="empty_scene",
                suggestion="No prose generated. Re-draft from the scene beat.",
            )],
        )

    # Length vs. target band — info-only, the LLM has nuance.
    word_count = len(_WORD.findall(text))
    target_words = int(scene_beat.get("target_words") or 600)
    floor = int(target_words * 0.65)
    ceiling = int(target_words * 1.45)
    if word_count < floor:
        findings.append(CriticFinding(
            critic="pacing", severity="warn", field="length",
            expected=f">={floor} words", observed=f"{word_count} words",
            suggestion="Scene may be rushed — extend the goal or expand the disaster.",
        ))
    elif word_count > ceiling:
        findings.append(CriticFinding(
            critic="pacing", severity="warn", field="length",
            expected=f"<={ceiling} words", observed=f"{word_count} words",
            suggestion="Scene may be padded — trim filler or split into two scenes.",
        ))

    # Hard topic-sentence / weather opener check — easy AI/amateur tell.
    opening = text.lstrip()[:200].lower()
    weather_opener = re.match(r"^(it was|it had been|the day was|the morning|the night was) ", opening)
    if weather_opener:
        findings.append(CriticFinding(
            critic="pacing", severity="warn", field="opening_hook",
            observed=opening[:60],
            suggestion="Open mid-moment — a body fact, an image, or a hard line.",
        ))

    # Closing-button check: very short final paragraph that sounds like a moral.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        final = paragraphs[-1]
        if len(_WORD.findall(final)) <= 12 and final.lower().startswith(("that was", "and so", "in the end", "thus")):
            findings.append(CriticFinding(
                critic="pacing", severity="warn", field="closing_pull",
                observed=final,
                suggestion="Replace the moralising button with a forward-pulling image or unresolved beat.",
            ))

    score = 1.0
    for f in findings:
        if f.severity == "error":
            score -= 0.25
        elif f.severity == "warn":
            score -= 0.10
        else:
            score -= 0.02
    score = max(0.0, round(score, 3))
    return CriticReport(
        critic="pacing",
        score=score,
        on_target=score >= 0.7,
        findings=findings,
    )


async def critique_pacing(
    *,
    draft: str,
    scene_beat: Dict[str, Any],
    router: Optional[Router] = None,
) -> CriticReport:
    heuristic = _heuristic_pacing(draft, scene_beat)

    user_prompt = build_critic_user_prompt(
        scene_beat_compact=compact_scene_beat(scene_beat),
        draft_prose=draft,
    )
    raw = await call_critic_llm(
        role="critic_pacing",
        system=PACING_CRITIC_SYSTEM,
        user_prompt=user_prompt,
        critic_name="pacing",
        router=router,
    )
    llm_report = parse_critic_report(raw, critic_name="pacing")
    return merge_reports(heuristic, llm_report)
