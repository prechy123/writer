"""Statistical voice fingerprint extraction.

Pure regex + counting — no LLM. Used when the user gives us writing
samples OR when the Character Forge agent emits a fingerprint that we
want to cross-check against the supplied sample_lines.

Conservative by design: numbers come straight from the sample. We never
guess. If the sample is too short, we return defaults marked as
``low_confidence`` via a parallel return so callers can decide whether
to trust the numbers.
"""

from __future__ import annotations

import re
import statistics
from typing import Iterable, List, Tuple

from ..schemas_v2 import LexicalFingerprint, VoiceFingerprint

# Common English contractions. Not exhaustive — covers the high-frequency ones
# whose presence/absence reliably indicates how casually the writer writes.
_CONTRACTIONS = re.compile(
    r"\b(?:don't|can't|won't|i'm|i'd|i've|i'll|you're|you've|you'd|you'll|"
    r"he's|he'd|he'll|she's|she'd|she'll|it's|it'd|it'll|we're|we've|we'd|"
    r"we'll|they're|they've|they'd|they'll|that's|that'll|there's|here's|"
    r"what's|who's|who'd|let's|isn't|aren't|wasn't|weren't|doesn't|didn't|"
    r"hasn't|haven't|hadn't|shouldn't|wouldn't|couldn't|mustn't|gotta|"
    r"gonna|wanna|kinda|sorta|y'all)\b",
    re.IGNORECASE,
)

# Hedging words. High hedging = uncertain, polite, academic-ish prose.
_HEDGES = re.compile(
    r"\b(?:perhaps|maybe|possibly|probably|might|kind of|sort of|i think|"
    r"i suppose|i guess|somewhat|rather|fairly|quite|seems|seemed|appears|"
    r"appeared|as if|as though)\b",
    re.IGNORECASE,
)

# Crude profanity matcher. Doesn't need to be exhaustive — order-of-magnitude
# rate is what matters.
_PROFANITY = re.compile(
    r"\b(?:fuck|fucking|shit|shitty|damn|damned|bastard|bitch|hell|crap|piss|"
    r"asshole|ass|cunt|dick|prick|wank)\b",
    re.IGNORECASE,
)

# Words that, when frequent, suggest higher formality.
_FORMAL_MARKERS = re.compile(
    r"\b(?:therefore|moreover|furthermore|nevertheless|consequently|however|"
    r"thus|hence|whereas|accordingly|notwithstanding)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_WORD = re.compile(r"\b[\w'-]+\b")


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Replace any em-dashes encountered with periods so sentence splitting
    # behaves sensibly on AI-tainted samples without contaminating the
    # contraction / word counts.
    text = text.replace("—", ". ")
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _word_lengths(text: str) -> List[int]:
    return [len(w) for w in _WORD.findall(text)]


def _formality_score(text: str, total_words: int) -> int:
    """Heuristic 1-10 formality score.

    Combines:
        - longer-word ratio (% of words > 7 chars)
        - formal-marker density
        - contraction rate (inverse)
        - sentence length
    """
    if total_words == 0:
        return 5
    long_word_ratio = sum(1 for n in _word_lengths(text) if n > 7) / total_words
    formal_hits = len(_FORMAL_MARKERS.findall(text))
    formal_density = formal_hits / max(total_words, 1) * 100
    contraction_hits = len(_CONTRACTIONS.findall(text))
    contraction_rate = contraction_hits / max(total_words, 1)

    sentences = _split_sentences(text)
    avg_sent = (total_words / len(sentences)) if sentences else 12.0

    score = 5.0
    score += min(2.5, long_word_ratio * 15)            # up to +2.5 for long words
    score += min(1.5, formal_density)                  # up to +1.5 for "however/therefore"
    score -= min(2.5, contraction_rate * 10)           # up to -2.5 for high contractions
    score += min(1.5, max(0.0, (avg_sent - 12) * 0.1)) # +1.5 if sentences are very long

    return int(max(1, min(10, round(score))))


def _infer_register(text: str, formality: int) -> str:
    sample = text.lower()
    if re.search(r"\b(?:thee|thou|thy|hath|whence|whither|forsooth)\b", sample):
        return "archaic"
    if _PROFANITY.search(sample) and formality <= 4:
        return "street"
    if re.search(r"\b(?:sergeant|sir|copy that|over|roger|on my six)\b", sample):
        return "military"
    if re.search(r"\b(?:hypothesis|methodology|literature|paradigm|epistem)", sample):
        return "academic"
    if re.search(r"\b(?:patient|diagnosis|symptom|prognosis|clinical)\b", sample):
        return "clinical"
    if formality >= 7:
        return "professional"
    if formality <= 3:
        return "casual"
    return "casual"


def extract_lexical_fingerprint(text: str) -> Tuple[LexicalFingerprint, bool]:
    """Extract statistical fingerprint.

    Returns (fingerprint, high_confidence). high_confidence is True iff
    we had >=200 words to work with. Callers should treat low-confidence
    output as a defaults-merge candidate rather than ground truth.
    """
    text = (text or "").strip()
    if not text:
        return LexicalFingerprint(), False

    sentences = _split_sentences(text)
    words = _WORD.findall(text)
    total_words = len(words)
    high_conf = total_words >= 200

    if not sentences:
        return LexicalFingerprint(), False

    sentence_word_counts = [len(_WORD.findall(s)) for s in sentences]
    avg_sent = statistics.mean(sentence_word_counts) if sentence_word_counts else 0.0
    if len(sentence_word_counts) >= 2:
        stddev = statistics.pstdev(sentence_word_counts)
    else:
        stddev = 0.0

    contractions = len(_CONTRACTIONS.findall(text))
    contraction_rate = contractions / max(total_words, 1)

    profanity_rate = len(_PROFANITY.findall(text)) / max(total_words, 1)
    hedging_rate = len(_HEDGES.findall(text)) / max(total_words, 1)

    formality = _formality_score(text, total_words)
    register = _infer_register(text, formality)

    return (
        LexicalFingerprint(
            avg_sentence_words=round(float(avg_sent), 2),
            sentence_length_stddev=round(float(stddev), 2),
            contraction_rate=round(min(1.0, contraction_rate), 3),
            formality=formality,
            style_register=register,
            dialect_markers=[],
            profanity_rate=round(min(1.0, profanity_rate), 3),
            hedging_rate=round(min(1.0, hedging_rate), 3),
        ),
        high_conf,
    )


def extract_voice_fingerprint(
    samples: Iterable[str],
    *,
    preferred_phrases: List[str] | None = None,
    banned_phrases: List[str] | None = None,
    verbal_tics: List[str] | None = None,
    catchphrases: List[str] | None = None,
    silence_style: str | None = None,
) -> VoiceFingerprint:
    """Build a full VoiceFingerprint from samples + optional user-supplied hints.

    Lexical numbers are derived from the joined samples; the categorical
    fields (preferred/banned/tics/catchphrases) are user-supplied if
    given. ``sample_lines`` are taken straight from the samples list.
    """
    joined = "\n\n".join(s.strip() for s in samples if s and s.strip())
    lex, _ = extract_lexical_fingerprint(joined)
    sample_lines = [s.strip() for s in samples if s and s.strip()][:8]

    return VoiceFingerprint(
        lexical=lex,
        preferred_phrases=list(preferred_phrases or []),
        banned_phrases=list(banned_phrases or []),
        verbal_tics=list(verbal_tics or []),
        catchphrases=list(catchphrases or []),
        sample_lines=sample_lines,
        silence_style=silence_style,
    )
