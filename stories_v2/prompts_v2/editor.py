"""Editor prompt.

Synthesises all critic findings into ONE coherent rewrite. The Editor
is NOT a writer-from-scratch; it edits a draft to fix the specific
issues the critic panel surfaced, while preserving everything that was
working.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .system_prelude import build_system

ROLE = """You are the Editor. A draft scene and a panel of critic findings are given. Your job: produce ONE revised version that addresses the findings while preserving everything that wasn't flagged.

PRINCIPLES:
- Do NOT rewrite the whole scene. Touch only what the findings call out.
- Where findings conflict, prefer Voice + Emotion findings over surface-style ones.
- Preserve the scene's structural choices (POV, opening hook, ending beat) unless a critic specifically flags them.
- Never lengthen the scene to "improve" it. If anything, trim.
- All the rules in the system prelude still apply: no em-dashes, no banned phrases, varied sentence length, contractions in dialogue, sensory anchors over abstract emotion.

OUTPUT FORMAT: prose only. No commentary, no headers, no diff markers. Just the revised scene start-to-finish."""

SYSTEM = build_system(ROLE)


def build_user_prompt(
    *,
    scene_beat_compact: str,
    draft_prose: str,
    critic_reports: List[Dict[str, Any]],
    overall_score: float,
) -> str:
    findings_serialised = json.dumps(
        [_compact_report(r) for r in critic_reports],
        indent=2,
        default=str,
    )
    return (
        "=== SCENE BEAT ===\n"
        + scene_beat_compact
        + "\n\n=== CURRENT DRAFT ===\n"
        + draft_prose
        + "\n\n=== CRITIC FINDINGS ===\n"
        + findings_serialised
        + f"\n\n=== OVERALL CRITIC SCORE ===\n{overall_score:.3f} (1.0 = perfect; 0.0 = total miss)"
        + "\n\n=== REWRITE THE SCENE NOW ===\n"
        "Apply only the specific fixes the critics requested. Keep what works. Prose only."
    )


def _compact_report(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "critic": report.get("critic"),
        "score": report.get("score"),
        "findings": [
            {
                "severity": f.get("severity"),
                "field": f.get("field"),
                "expected": f.get("expected"),
                "observed": f.get("observed"),
                "span": f.get("span"),
                "suggestion": f.get("suggestion"),
            }
            for f in (report.get("findings") or [])
            if f.get("severity") != "info" or f.get("suggestion")
        ],
    }
