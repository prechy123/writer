"""Validate generated dialogue against a character's voice fingerprint.

Pure-Python scoring; the Voice Critic agent supplements this with LLM
qualitative judgement. Output is a (score, findings[]) tuple where:
    score is in [0, 1]   — 1.0 is a perfect match
    findings is a list of structured deviations the Editor can fix
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import List

from ..schemas_v2 import LexicalFingerprint, VoiceFingerprint
from .extractor import _CONTRACTIONS, _HEDGES, _PROFANITY, _WORD, _split_sentences


@dataclass
class VoiceFinding:
    severity: str  # "info" | "warn" | "error"
    field: str
    expected: str
    observed: str
    note: str = ""


@dataclass
class VoiceReport:
    score: float
    findings: List[VoiceFinding] = field(default_factory=list)


def score_dialogue_match(
    dialogue: str,
    fingerprint: VoiceFingerprint,
) -> VoiceReport:
    """Score a chunk of dialogue against a voice fingerprint.

    The score is a soft cosine-style distance: each axis contributes a
    bounded penalty, max penalty per axis is configurable. The final
    score is ``1 - sum(penalties)`` clamped to [0, 1].

    The Voice Critic agent gets this score + findings; if score < 0.6
    the Editor is asked to rewrite the dialogue using the fingerprint's
    sample_lines as anchors.
    """
    findings: List[VoiceFinding] = []
    if not dialogue or not dialogue.strip():
        return VoiceReport(score=1.0)

    text = dialogue
    sentences = _split_sentences(text)
    words = _WORD.findall(text)
    total_words = len(words)
    if total_words == 0:
        return VoiceReport(score=1.0)

    lex: LexicalFingerprint = fingerprint.lexical
    penalty = 0.0

    # --- contraction rate ---
    obs_contraction_rate = len(_CONTRACTIONS.findall(text)) / total_words
    diff = abs(obs_contraction_rate - lex.contraction_rate)
    if diff > 0.15:
        sev = "warn" if diff < 0.30 else "error"
        penalty += min(0.20, diff)
        findings.append(
            VoiceFinding(
                severity=sev,
                field="contraction_rate",
                expected=f"{lex.contraction_rate:.2f}",
                observed=f"{obs_contraction_rate:.2f}",
                note="Use more contractions to match the character's casual register."
                if obs_contraction_rate < lex.contraction_rate
                else "This character contracts less than the prose suggests.",
            )
        )

    # --- sentence length avg ---
    if sentences:
        sentence_lens = [len(_WORD.findall(s)) for s in sentences]
        obs_avg = statistics.mean(sentence_lens)
        avg_diff = abs(obs_avg - lex.avg_sentence_words) / max(lex.avg_sentence_words, 1.0)
        if avg_diff > 0.4:
            sev = "warn" if avg_diff < 0.8 else "error"
            penalty += min(0.15, avg_diff * 0.15)
            findings.append(
                VoiceFinding(
                    severity=sev,
                    field="avg_sentence_words",
                    expected=f"{lex.avg_sentence_words:.1f}",
                    observed=f"{obs_avg:.1f}",
                    note="Sentence length is off — split or merge to match the character.",
                )
            )

        if len(sentence_lens) >= 2:
            obs_stddev = statistics.pstdev(sentence_lens)
            if obs_stddev < 4.0:
                penalty += 0.15
                findings.append(
                    VoiceFinding(
                        severity="warn",
                        field="sentence_length_stddev",
                        expected=">8.0",
                        observed=f"{obs_stddev:.1f}",
                        note="Sentences are too uniform — vary length wildly.",
                    )
                )

    # --- hedging ---
    obs_hedge_rate = len(_HEDGES.findall(text)) / total_words
    if obs_hedge_rate > lex.hedging_rate + 0.03:
        penalty += 0.08
        findings.append(
            VoiceFinding(
                severity="warn",
                field="hedging_rate",
                expected=f"{lex.hedging_rate:.2f}",
                observed=f"{obs_hedge_rate:.2f}",
                note="Too much hedging — this character is more decisive.",
            )
        )

    # --- profanity ---
    obs_prof_rate = len(_PROFANITY.findall(text)) / total_words
    if obs_prof_rate > lex.profanity_rate + 0.02:
        penalty += 0.05
        findings.append(
            VoiceFinding(
                severity="info",
                field="profanity_rate",
                expected=f"{lex.profanity_rate:.2f}",
                observed=f"{obs_prof_rate:.2f}",
                note="Profanity above the character's baseline.",
            )
        )

    # --- banned phrases ---
    lowered = text.lower()
    for phrase in fingerprint.banned_phrases:
        if phrase and phrase.lower() in lowered:
            penalty += 0.20
            findings.append(
                VoiceFinding(
                    severity="error",
                    field="banned_phrases",
                    expected="absent",
                    observed=phrase,
                    note=f"Character would NEVER say {phrase!r}.",
                )
            )

    # --- preferred-phrase presence (informational only — absence isn't punished) ---
    for phrase in fingerprint.preferred_phrases:
        if phrase and phrase.lower() in lowered:
            findings.append(
                VoiceFinding(
                    severity="info",
                    field="preferred_phrases",
                    expected="present",
                    observed=phrase,
                    note=f"Used the character's signature {phrase!r}.",
                )
            )
            break  # one note is enough; don't spam

    score = max(0.0, min(1.0, 1.0 - penalty))
    return VoiceReport(score=round(score, 3), findings=findings)


def validate_voice(dialogue: str, fingerprint: VoiceFingerprint) -> VoiceReport:
    """Convenience alias for clarity at the agent call site."""
    return score_dialogue_match(dialogue, fingerprint)
