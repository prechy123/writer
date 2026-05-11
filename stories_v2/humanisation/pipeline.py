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
from . import banned_tokens, burstiness, contractions, detector_gate, fragments, idiom_inject

logger = logging.getLogger(__name__)


async def humanise(
    prose: str,
    *,
    present_characters: Optional[List[CharacterBibleV2]] = None,
    run_detector: bool = False,
) -> tuple[str, HumanisationReport]:
    """Apply the full deterministic humanisation pass.

    Returns ``(cleaned_prose, HumanisationReport)``. Pure-Python except
    for the optional external detector hop.
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

    # 4. Burstiness enforcement (conservative)
    prose, splits_made = burstiness.enforce(prose)

    # 5. Final measurements
    post_mean, post_stddev, post_count = burstiness.measure(prose)

    # 6. Fragment count
    frag_count, total_sentences = fragments.count_fragments(prose)
    if total_sentences >= 6 and frag_count == 0:
        notes.append("no_sentence_fragments_present")

    # 7. Idiom audit (no rewrite)
    if present_characters:
        notes.extend(idiom_inject.audit(prose, present_characters))

    # 8. Optional detector
    detector_score: Optional[float] = None
    if run_detector:
        detector_score = await detector_gate.score(prose)

    report = HumanisationReport(
        em_dash_replacements=em_count,
        banned_phrase_strikes=strikes,
        contractions_injected=contractions_count,
        fragments_injected=0,  # fragments module measures only
        sentences_split=splits_made,
        sentences_merged=0,
        burstiness_before=round(pre_stddev, 2),
        burstiness_after=round(post_stddev, 2),
        detector_score_before=None,
        detector_score_after=detector_score,
        notes=notes,
    )
    return prose, report
