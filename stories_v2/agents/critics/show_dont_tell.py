"""Show-don't-tell critic — regex first, LLM second.

Regex catches the unambiguous "X felt Y" / "X was Y" patterns; the LLM
catches the subtler authorial summaries the regex can't pick.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ...prompts_v2.critics import SHOW_DONT_TELL_CRITIC_SYSTEM, build_critic_user_prompt
from ...providers import Router
from ...schemas_v2 import CriticFinding, CriticReport
from ._common import call_critic_llm, compact_scene_beat, merge_reports, parse_critic_report

# Emotion words frequently used in flat "told" prose
_EMOTION_WORDS = (
    "happy|sad|angry|furious|scared|terrified|nervous|anxious|"
    "excited|joyful|miserable|melancholy|frustrated|annoyed|"
    "calm|relieved|shocked|surprised|confused|bewildered|"
    "tense|uneasy|content|peaceful|distressed|devastated|jealous|"
    "grief|fear|joy|sadness|anger|sorrow|dread|panic|elation|despair|"
    "rage|hope|hopelessness|guilt|shame|pride|envy|longing|relief"
)

# Outside-of-dialogue narrator-tell patterns. We avoid matching inside
# quoted dialogue by checking the line doesn't sit inside double quotes —
# done by the orchestrator's split, not here, to keep regex simple.
_NARR_FELT = re.compile(
    rf"\b([A-Z][\w]+)\s+(felt|was)\s+({_EMOTION_WORDS})\b",
    re.IGNORECASE,
)
_WAVE_OF = re.compile(
    rf"\ba\s+(wave|surge|wash|flood|rush)\s+of\s+({_EMOTION_WORDS})\b",
    re.IGNORECASE,
)
_SENSE_OF = re.compile(
    rf"\b(felt|had)\s+a\s+sense\s+of\s+({_EMOTION_WORDS})\b",
    re.IGNORECASE,
)
_ADVERB_TAG = re.compile(
    r"\b(said|asked|replied|whispered|murmured|muttered|shouted|cried)\s+\w+ly\b",
    re.IGNORECASE,
)
_TOLD_ATMOSPHERE = re.compile(
    r"\b(the (room|silence|air|atmosphere|mood))\s+(was|felt|seemed)\s+(tense|thick|heavy|electric|charged)\b",
    re.IGNORECASE,
)


def _heuristic_show_dont_tell(draft: str) -> CriticReport:
    findings: List[CriticFinding] = []

    def _add(severity: str, label: str, span: str, suggestion: str) -> None:
        findings.append(
            CriticFinding(
                critic="show_dont_tell",
                severity=severity,  # type: ignore[arg-type]
                field=label,
                span=span,
                suggestion=suggestion,
                rationale="Narrator tells an emotion the reader could be shown.",
            )
        )

    for m in _NARR_FELT.finditer(draft):
        _add("warn", "narrator_felt", m.group(0),
             "Show via body, breath, action, or silence instead.")
    for m in _WAVE_OF.finditer(draft):
        _add("warn", "wave_of_emotion", m.group(0),
             "Replace the cliché construction with a specific physical detail.")
    for m in _SENSE_OF.finditer(draft):
        _add("warn", "sense_of_emotion", m.group(0),
             "Pick a concrete observation that creates the same sense in the reader.")
    for m in _ADVERB_TAG.finditer(draft):
        _add("info", "adverb_dialogue_tag", m.group(0),
             "Move the emotion into a body beat between dialogue lines.")
    for m in _TOLD_ATMOSPHERE.finditer(draft):
        _add("warn", "told_atmosphere", m.group(0),
             "Show the tension through what someone does, smells, hears, or doesn't say.")

    # Score: small linear penalty per hit, capped.
    penalty = min(0.5, 0.07 * len(findings))
    score = round(1.0 - penalty, 3)
    return CriticReport(
        critic="show_dont_tell",
        score=score,
        on_target=score >= 0.7,
        findings=findings,
    )


async def critique_show_dont_tell(
    *,
    draft: str,
    scene_beat: Dict[str, Any],
    router: Optional[Router] = None,
) -> CriticReport:
    heuristic = _heuristic_show_dont_tell(draft)

    user_prompt = build_critic_user_prompt(
        scene_beat_compact=compact_scene_beat(scene_beat),
        draft_prose=draft,
    )
    raw = await call_critic_llm(
        role="critic_show_dont_tell",
        system=SHOW_DONT_TELL_CRITIC_SYSTEM,
        user_prompt=user_prompt,
        critic_name="show_dont_tell",
        router=router,
    )
    llm_report = parse_critic_report(raw, critic_name="show_dont_tell")
    return merge_reports(heuristic, llm_report)
