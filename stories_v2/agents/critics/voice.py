"""Voice critic — hybrid: statistical validator + LLM qualitative judgement."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...prompts_v2.critics import VOICE_CRITIC_SYSTEM, build_critic_user_prompt
from ...providers import Router
from ...schemas_v2 import CharacterBibleV2, CriticFinding, CriticReport
from ...voice import score_dialogue_match
from ._common import (
    call_critic_llm,
    compact_scene_beat,
    extract_dialogue,
    merge_reports,
    parse_critic_report,
)

logger = logging.getLogger(__name__)


async def critique_voice(
    *,
    draft: str,
    scene_beat: Dict[str, Any],
    present_characters: List[CharacterBibleV2],
    router: Optional[Router] = None,
) -> CriticReport:
    """Combine the deterministic dialogue-fingerprint check with the LLM
    qualitative critic. The two scores merge (min wins)."""
    heuristic = _heuristic_voice_check(draft, present_characters)

    user_prompt = build_critic_user_prompt(
        scene_beat_compact=compact_scene_beat(scene_beat),
        draft_prose=draft,
    )
    # Pass per-character fingerprint summaries so the LLM has context.
    fingerprint_summary = _fingerprints_summary(present_characters)
    if fingerprint_summary:
        user_prompt = (
            "=== CHARACTER FINGERPRINTS ===\n"
            + fingerprint_summary
            + "\n\n"
            + user_prompt
        )

    raw = await call_critic_llm(
        role="critic_voice",
        system=VOICE_CRITIC_SYSTEM,
        user_prompt=user_prompt,
        critic_name="voice",
        router=router,
    )
    llm_report = parse_critic_report(raw, critic_name="voice")

    return merge_reports(heuristic, llm_report)


# ---------------------------------------------------------------------------

def _heuristic_voice_check(
    draft: str,
    present_characters: List[CharacterBibleV2],
) -> CriticReport:
    """Validate ALL extracted dialogue against the fingerprint of each
    main/recurring character with a fingerprint. We can't perfectly
    attribute lines to speakers without an LLM pass, so we apply the
    most-lenient fingerprint across the chorus — that catches the
    universal violations (em-dashes, banned phrases, hard register
    mismatches) without false-positives on speaker attribution.
    """
    dialogue = extract_dialogue(draft)
    if not dialogue:
        return CriticReport(critic="voice", score=1.0, on_target=True)

    joined = " ".join(dialogue)

    fingerprinted = [
        c for c in present_characters
        if c.voice_fingerprint is not None
    ]
    if not fingerprinted:
        return CriticReport(critic="voice", score=1.0, on_target=True)

    findings: List[CriticFinding] = []
    worst_score = 1.0
    for char in fingerprinted:
        report = score_dialogue_match(joined, char.voice_fingerprint)
        if report.score < worst_score:
            worst_score = report.score
        for f in report.findings:
            # Tag findings with the character whose fingerprint flagged them.
            findings.append(
                CriticFinding(
                    critic="voice",
                    severity=f.severity if f.severity in ("info", "warn", "error") else "info",
                    field=f.field,
                    expected=f.expected,
                    observed=f.observed,
                    suggestion=f.note or None,
                    rationale=f"flagged against fingerprint of {char.name}",
                )
            )
    return CriticReport(
        critic="voice",
        score=round(worst_score, 3),
        on_target=worst_score >= 0.7,
        findings=findings,
    )


def _fingerprints_summary(chars: List[CharacterBibleV2]) -> str:
    """Compact 'who speaks how' block for the LLM critic."""
    rows = []
    for c in chars:
        if not c.voice_fingerprint:
            continue
        vf = c.voice_fingerprint
        bits = [f"{c.name}"]
        lex = vf.lexical
        bits.append(
            f"avg={lex.avg_sentence_words:.0f}w stddev={lex.sentence_length_stddev:.0f}"
        )
        bits.append(f"contract={lex.contraction_rate:.2f} formality={lex.formality}/10 register={lex.style_register}")
        if vf.sample_lines:
            sample = vf.sample_lines[0]
            bits.append(f'sample: "{sample}"')
        if vf.banned_phrases:
            bits.append("never_says: " + "; ".join(vf.banned_phrases[:3]))
        rows.append(" | ".join(bits))
    return "\n".join(rows)
