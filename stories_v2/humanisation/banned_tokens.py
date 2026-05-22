"""Em-dash strip + AI-phrase blocklist.

This module never asks the LLM. It does the most important work in the
v2 stack: removes the single signal that AI detectors weight most
heavily.

We replace em-dashes contextually:
  - Mid-sentence interruption  → period + capitalise the next word
  - Parenthetical aside        → commas
  - Compound break             → semicolon NOT used (also a tell);
                                  prefer period + capital
"""

from __future__ import annotations

import re
from typing import List, Tuple


# All variants of the em-dash glyph, plus the rarer minus / quotation dash.
_DASH_RE = re.compile(r"[—―−⸺⸻]")

# Word-character followed by dash followed by word-character — likely a
# mid-sentence interruption that wants a period+capital.
_INTERRUPT_RE = re.compile(r"(\w)[—―−⸺⸻](\w)")

# Sentence-end dash glyph followed by " word" — usually wants a period+capital.
_SENTENCE_END_RE = re.compile(r"([\.!\?])[\s]*[—―−⸺⸻][\s]+(\w)")

# Phrase blocklist. Conservative substitutions chosen so deletion never leaves
# a sentence dangling. Where no clean substitute exists, the word becomes the
# empty string (the surrounding context still reads).
PHRASE_BLOCKLIST: List[Tuple[re.Pattern, str]] = [
    # --- Safe swaps: replacement keeps the sentence grammatical ---
    (re.compile(r"\b(delves?|delving|delved)\s+into\b", re.IGNORECASE), "looks into"),
    (re.compile(r"\bnavigate(?:s|d|ing)?\s+the\s+complexit(?:y|ies)\s+of\b", re.IGNORECASE), "work through"),
    (re.compile(r"\bunwavering\b", re.IGNORECASE), "steady"),
    (re.compile(r"\bis\s+a\s+testament\s+to\b", re.IGNORECASE), "shows"),
    (re.compile(r"\bmyriad\b", re.IGNORECASE), "many"),
    (re.compile(r"\bbustling\b", re.IGNORECASE), "busy"),
    (re.compile(r"\bnestled\b", re.IGNORECASE), "tucked"),
    (re.compile(r"\bgleaming\b", re.IGNORECASE), "bright"),
    (re.compile(r"\bpristine\b", re.IGNORECASE), "clean"),
    (re.compile(r"\bever\s+so\s+slightly\b", re.IGNORECASE), "slightly"),
    (re.compile(r"\bwet\s+stone\b", re.IGNORECASE), "stone"),
    # Reduce "a low/thin/soft growl" → "a growl" — keep the noun, drop padding.
    (re.compile(r"\ba\s+(?:low|soft|thin|sharp|distant|faint)\s+(hum|growl|thrum|whine|moan|groan|roar)\b", re.IGNORECASE), r"a \1"),

    # --- Safe deletions: removing leaves the rest of the sentence intact ---
    (re.compile(r"\b(tapestry of\s+\w+)\b", re.IGNORECASE), ""),
    (re.compile(r"\b(it\'s\s+important\s+to\s+note\s+that|in\s+conclusion,?|at\s+its\s+core,?)\b", re.IGNORECASE), ""),
    (re.compile(r"\bin\s+stark\s+contrast(?:\s+to)?\s*", re.IGNORECASE), ""),
    (re.compile(r",?\s*to\s+no\s+one\s+in\s+particular", re.IGNORECASE), ""),
    (re.compile(r"\b(?:moments?|seconds?)\s+later,?\s+", re.IGNORECASE), ""),
]


# Sentence killers: phrases that ARE the verb spine of the sentence they
# appear in. Removing the phrase leaves a stub; the whole sentence has
# to be dropped. Each pattern is matched against a sentence; if it
# matches, the entire sentence is removed from the output.
SENTENCE_KILLERS: List[re.Pattern] = [
    # "He let out a held breath." / "She let out a long sigh."
    re.compile(r"\b(?:let|let\s+out|exhaled)\s+(?:a|an)\s+(?:held|long|slow|shallow|ragged|sharp|deep)\s+(?:breath|sigh|exhale)\b", re.IGNORECASE),
    # "He took a deep breath."
    re.compile(r"\btook\s+a\s+(?:deep|long|slow|sharp|ragged)\s+breath\b", re.IGNORECASE),
    # "He couldn't help but stare." — strip whole sentence.
    re.compile(r"\b(?:couldn\'?t|could\s+not)\s+help\s+but\s+", re.IGNORECASE),
    # "He found himself running."
    re.compile(r"\bfound\s+(?:himself|herself|themselves|itself)\s+", re.IGNORECASE),
    # "A wave of fear washed over him."
    re.compile(r"\ba\s+(?:wave|flood|surge|rush|wash|tide)\s+of\s+\w+\s+(?:washed|swept|crashed|rolled|flowed)\s+over\s+\w+", re.IGNORECASE),
    # "The air was thick with tension."
    re.compile(r"\bthe\s+air\s+(?:was|grew|hung|turned|felt)\s+thick\s+with\s+", re.IGNORECASE),
    # "thick with tension" / "thick with dread" — telling.
    re.compile(r"\bthick\s+with\s+(?:tension|silence|fear|dread|anger|sorrow|grief)\b", re.IGNORECASE),
    # "A thin halo of blue light surrounded the rune."
    re.compile(r"\ba\s+(?:thin|soft|faint|bright)\s+halo\s+of\s+\w+\s+light\b", re.IGNORECASE),
    # "The metallic tang of ozone..."
    re.compile(r"\b(?:a|the)\s+metallic\s+tang\b", re.IGNORECASE),
    # "...echoed off the walls"
    re.compile(r"\b(?:echoed|reverberated)\s+(?:off|through|across|down)\s+the\s+\w+", re.IGNORECASE),
    # "in time with his heartbeat"
    re.compile(r"\bin\s+time\s+with\s+(?:his|her|their|the|its)\s+(?:heartbeat|pulse|breath)\b", re.IGNORECASE),
    # "Cold steel bit his palm."
    re.compile(r"\b(?:cold|hot|sharp)\s+(?:metal|steel)\s+bit(?:ing|e)\s+(?:his|her|their|its)\s+\w+", re.IGNORECASE),
    # "casting jagged shadows on the wall"
    re.compile(r"\bcasting\s+\w+\s+shadows?\s+(?:on|across|over)\s+", re.IGNORECASE),
    # "a sense of dread" — telling.
    re.compile(r"\ba\s+sense\s+of\s+(?:dread|unease|foreboding|calm|peace|loss|wonder)\b", re.IGNORECASE),
    # "the weight of his decision" — generic abstraction.
    re.compile(r"\bthe\s+weight\s+of\s+(?:his|her|their|the|that)\s+(?:decision|words|silence|grief|loss|past)\b", re.IGNORECASE),
]


def strip_em_dashes(text: str) -> Tuple[str, int]:
    """Return (cleaned_text, replacements_made)."""
    if not text or "—" not in text and not _DASH_RE.search(text):
        return text, 0

    count = len(_DASH_RE.findall(text))

    # Pattern 1: sentence-terminator + dash + capital -> drop the dash, keep period
    text = _SENTENCE_END_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}", text)

    # Pattern 2: word—word -> word. Word
    def _interrupt(m: re.Match) -> str:
        return f"{m.group(1)}. {m.group(2).upper()}"

    text = _INTERRUPT_RE.sub(_interrupt, text)

    # Any leftover dashes — replace with a simple comma fallback. (Some occur
    # in lists or attribution; comma is the safest universal substitute.)
    text = _DASH_RE.sub(", ", text)

    # Collapse any double-spaces we introduced.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text, count


def strip_blocklist(text: str) -> Tuple[str, int]:
    """Apply safe phrase swaps + drop whole sentences containing killer phrases.

    Returns (cleaned, strikes_count). Two-pass:
      1. Drop sentences matching SENTENCE_KILLERS entirely.
      2. Run the safe PHRASE_BLOCKLIST substitutions on what remains.
      3. Tidy whitespace + drop residual degenerate stubs.
    """
    if not text:
        return text, 0
    strikes = 0

    # ---- Pass 1: kill whole sentences for verb-spine phrases ----
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept_sentences: List[str] = []
    for s in sentences:
        killed = False
        for pat in SENTENCE_KILLERS:
            if pat.search(s):
                killed = True
                strikes += 1
                break
        if not killed:
            kept_sentences.append(s)
    text = " ".join(kept_sentences)

    # ---- Pass 2: safe in-place swaps ----
    for pattern, replacement in PHRASE_BLOCKLIST:
        new_text, n = pattern.subn(replacement, text)
        if n > 0:
            strikes += n
            text = new_text

    # ---- Pass 3: cleanup artefacts and any residual degenerate sentences ----
    text = re.sub(r"\s+([,.;:!\?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\.\s*\.+", ".", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"(^|[.!?]\s+),\s*", r"\1", text)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept: List[str] = []
    for s in sentences:
        if _is_degenerate_sentence(s):
            continue
        kept.append(s)
    text = " ".join(kept).strip()
    return text, strikes


_DEGENERATE_HEADS = re.compile(
    r"^\s*(?:He|She|They|It|I|We|You|His|Her|Their|My|Our|Your)"
    r"(?:\s+(?:was|were|is|are|had|has|have|did|do|does|would|could|should|will|might))?\s*[.!?]?\s*$",
    re.IGNORECASE,
)


def _is_degenerate_sentence(sentence: str) -> bool:
    """A sentence is degenerate when it's just a pronoun + maybe an aux verb
    with no actual content, or when it's < 3 word tokens total."""
    if not sentence:
        return True
    stripped = sentence.strip()
    if not stripped:
        return True
    # Bare pronoun-only or pronoun+aux only.
    if _DEGENERATE_HEADS.match(stripped):
        return True
    word_count = len(re.findall(r"[A-Za-z']+", stripped))
    if word_count < 3:
        # Short sentences are legitimate fragments — keep if they end with
        # punctuation suggesting intent. Drop only the truly empty.
        return word_count == 0
    return False
