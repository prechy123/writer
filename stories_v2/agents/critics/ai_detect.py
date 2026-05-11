"""AI-detect critic — surfaces patterns Phase 7's deterministic strip
won't catch (rhetorical-bow endings, uniform structures, hedging
saturation, dialogue-tag adverbs).
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List, Optional

from ...prompts_v2.critics import AI_DETECT_CRITIC_SYSTEM, build_critic_user_prompt
from ...providers import Router
from ...schemas_v2 import CriticFinding, CriticReport
from ._common import call_critic_llm, compact_scene_beat, merge_reports, parse_critic_report

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_WORD = re.compile(r"\b[\w'-]+\b")

# Em-dashes shouldn't survive Phase 7, but in case the writer slipped one
# in we still flag here for visibility.
_EM_DASH_RE = re.compile(r"—")

# Telltale "it's important to note" / "in conclusion" / "at its core"
# already in the deterministic strip, but we flag here too so the editor
# rewrites instead of leaving a hole.
_AI_PHRASE_RE = re.compile(
    r"\b(?:delve|tapestry|navigate the complexities|unwavering|"
    r"testament to|in conclusion|it's important to note|at its core|"
    r"myriad|bustling|nestled|gleaming|pristine)\b",
    re.IGNORECASE,
)

_HEDGE_RE = re.compile(
    r"\b(?:perhaps|maybe|seems|seemed|appears|appeared|as if|as though|"
    r"somewhat|rather|fairly|quite|kind of|sort of)\b",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def _heuristic_ai_detect(draft: str) -> CriticReport:
    findings: List[CriticFinding] = []

    def _add(severity: str, field: str, span: str, suggestion: str) -> None:
        findings.append(
            CriticFinding(
                critic="ai_detect",
                severity=severity,  # type: ignore[arg-type]
                field=field,
                span=span,
                suggestion=suggestion,
                rationale="Pattern correlates with LLM-generated prose.",
            )
        )

    sentences = _split_sentences(draft)
    word_counts = [len(_WORD.findall(s)) for s in sentences]
    total_words = sum(word_counts)

    # Burstiness — low sentence-length variance is a strong AI signal.
    if len(word_counts) >= 4:
        stddev = statistics.pstdev(word_counts)
        if stddev < 4.0 and total_words > 200:
            _add("warn", "low_burstiness",
                 f"avg={statistics.mean(word_counts):.1f} stddev={stddev:.1f}",
                 "Mix three-word fragments with twenty-word sentences. Vary length wildly.")

    # Em-dashes (last line of defence before Phase 7)
    em_dash_count = len(_EM_DASH_RE.findall(draft))
    if em_dash_count:
        _add("error", "em_dash_present",
             f"{em_dash_count} em-dash{'es' if em_dash_count > 1 else ''}",
             "Replace every em-dash with a period+capital, comma, or double-hyphen.")

    # Banned AI phrases
    for m in _AI_PHRASE_RE.finditer(draft):
        _add("error", "banned_ai_phrase", m.group(0),
             "Strike this phrase. Rewrite with concrete language.")

    # Excessive hedging
    if total_words >= 200:
        hedge_count = len(_HEDGE_RE.findall(draft))
        hedge_rate = hedge_count / total_words
        if hedge_rate > 0.025:
            _add("warn", "hedging_saturation", f"{hedge_count} hedges in {total_words} words",
                 "Strip half the hedges. Most can become declarative.")

    # All-paragraphs-start-with-noun-phrase check
    paragraphs = [p for p in draft.split("\n\n") if p.strip()]
    noun_starts = 0
    pronoun_starts = 0
    for p in paragraphs:
        first = p.lstrip().split(" ", 1)[0].strip(',."').lower()
        if first in {"he", "she", "they", "it", "i"}:
            pronoun_starts += 1
        elif first.istitle() and first not in {"And", "But", "So", "Then", "Yet"}:
            noun_starts += 1
    if len(paragraphs) >= 4 and noun_starts / len(paragraphs) > 0.75:
        _add("warn", "uniform_paragraph_starts",
             f"{noun_starts}/{len(paragraphs)} paragraphs start with a noun",
             "Open some paragraphs with action, dialogue, fragments, or a body detail.")

    # Score
    score = 1.0
    for f in findings:
        if f.severity == "error":
            score -= 0.15
        elif f.severity == "warn":
            score -= 0.07
    score = max(0.0, round(score, 3))

    return CriticReport(
        critic="ai_detect",
        score=score,
        on_target=score >= 0.7,
        findings=findings,
    )


async def critique_ai_detect(
    *,
    draft: str,
    scene_beat: Dict[str, Any],
    router: Optional[Router] = None,
) -> CriticReport:
    heuristic = _heuristic_ai_detect(draft)

    user_prompt = build_critic_user_prompt(
        scene_beat_compact=compact_scene_beat(scene_beat),
        draft_prose=draft,
    )
    raw = await call_critic_llm(
        role="critic_ai_detect",
        system=AI_DETECT_CRITIC_SYSTEM,
        user_prompt=user_prompt,
        critic_name="ai_detect",
        router=router,
    )
    llm_report = parse_critic_report(raw, critic_name="ai_detect")
    return merge_reports(heuristic, llm_report)
