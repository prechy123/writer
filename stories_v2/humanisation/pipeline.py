"""The humanisation pipeline — single entry point.

Sequence (in order):
    1. strip_em_dashes
    2. strip_blocklist
    3. inject_in_dialogue       (contractions)
    4. burstiness.enforce       (conservative splits)
    5. burstiness.measure       (final stddev report)
    6. fragments.count
    7. idiom_inject.audit       (notes only — no rewrite)
    8. detector_gate.score      (optional external)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ..schemas_v2 import CharacterBibleV2, HumanisationReport
from . import banned_tokens, burstiness, contractions, detector_gate, fragments, idiom_inject, repetition

logger = logging.getLogger(__name__)


async def humanise(
    prose: str,
    *,
    present_characters: Optional[List[CharacterBibleV2]] = None,
    run_detector: bool = False,
    prior_scenes_prose: Optional[List[str]] = None,
) -> tuple[str, HumanisationReport]:
    """Apply the full deterministic humanisation pass.

    Returns ``(cleaned_prose, HumanisationReport)``. Pure-Python except
    for the optional external detector hop.

    ``prior_scenes_prose`` enables cross-scene repetition stripping.
    Pass the committed prose of recent scenes; the pipeline will detect
    n-gram phrases that repeat across multiple prior scenes and drop the
    sentences anchored on them.
    """
    if not prose:
        return prose, HumanisationReport()

    pre_mean, pre_stddev, pre_count = burstiness.measure(prose)
    notes: List[str] = []

    # 1. Em-dashes
    prose, em_count = banned_tokens.strip_em_dashes(prose)

    # 2. Banned phrases
    prose, strikes = banned_tokens.strip_blocklist(prose)

    # 3. Contractions in dialogue
    prose, contractions_count = contractions.inject_in_dialogue(prose)

    # 4. Cross-scene repetition strip
    repeats_dropped = 0
    if prior_scenes_prose:
        prior_index = repetition.build_prior_phrase_index(prior_scenes_prose)
        repeats = repetition.find_repeats(prose, prior_index=prior_index)
        if repeats:
            # Note up to 5 of the worst repeats so the editor can see them next cycle.
            for phrase, count in repeats[:5]:
                notes.append(f"repeat_phrase[{count}x]:{phrase}")
            prose, repeats_dropped = repetition.strip_obvious_repeats(prose, repeats=repeats)

    # 5. Burstiness enforcement (conservative)
    prose, splits_made = burstiness.enforce(prose)

    # 6. Final measurements
    post_mean, post_stddev, post_count = burstiness.measure(prose)

    # 7. Fragment count
    frag_count, total_sentences = fragments.count_fragments(prose)
    if total_sentences >= 6 and frag_count == 0:
        notes.append("no_sentence_fragments_present")

    # 8. Idiom audit (no rewrite)
    if present_characters:
        notes.extend(idiom_inject.audit(prose, present_characters))

    # 9. Optional detector
    detector_score: Optional[float] = None
    if run_detector:
        detector_score = await detector_gate.score(prose)

    if repeats_dropped:
        notes.append(f"repeated_sentences_dropped:{repeats_dropped}")

    report = HumanisationReport(
        em_dash_replacements=em_count,
        banned_phrase_strikes=strikes,
        contractions_injected=contractions_count,
        fragments_injected=0,  # fragments module measures only
        sentences_split=splits_made,
        sentences_merged=repeats_dropped,
        burstiness_before=round(pre_stddev, 2),
        burstiness_after=round(post_stddev, 2),
        detector_score_before=None,
        detector_score_after=detector_score,
        notes=notes,
    )
    return prose, report
