"""Shared utilities for the critic agents."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ...providers import Router, get_router
from ...schemas_v2 import CriticFinding, CriticName, CriticReport, Severity

logger = logging.getLogger(__name__)


def compact_scene_beat(scene_beat: Dict[str, Any]) -> str:
    """Trim the scene beat to what critics need."""
    keep = (
        "scene_idx", "title", "summary", "kisho_phase",
        "pov_character_name", "location", "time_of_day",
        "goal", "conflict", "disaster",
        "protagonist_start_emotion", "protagonist_end_emotion",
        "reader_start_emotion", "reader_end_emotion",
        "sensory_focus", "interiority_density", "techniques",
        "target_words",
    )
    sub = {k: scene_beat.get(k) for k in keep if scene_beat.get(k) is not None}
    return json.dumps(sub, indent=2, default=str)


async def call_critic_llm(
    *,
    role: str,
    system: str,
    user_prompt: str,
    critic_name: CriticName,
    router: Optional[Router] = None,
    max_tokens: int = 1500,
    temperature: float = 0.2,
) -> Optional[Dict[str, Any]]:
    """Issue a critic LLM call, return parsed dict or None on failure."""
    router = router or get_router()
    try:
        return await router.chat_json(
            role=role,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=90.0,
        )
    except Exception as exc:
        logger.warning("critic[%s]: LLM call failed (%s)", critic_name, exc)
        return None


def parse_critic_report(
    raw: Optional[Dict[str, Any]],
    *,
    critic_name: CriticName,
    fallback_score: float = 1.0,
) -> CriticReport:
    """Validate the LLM's JSON into a CriticReport, with safe defaults."""
    if not isinstance(raw, dict):
        return CriticReport(critic=critic_name, score=fallback_score, on_target=True, findings=[])

    score = _clamp_score(raw.get("score", fallback_score))
    on_target = bool(raw.get("on_target", score >= 0.7))
    findings_raw = raw.get("findings") or []
    findings: List[CriticFinding] = []
    for entry in findings_raw:
        if not isinstance(entry, dict):
            continue
        entry["critic"] = critic_name  # enforce
        sev = entry.get("severity") or "info"
        if sev not in ("info", "warn", "error"):
            sev = "info"
        entry["severity"] = sev
        try:
            findings.append(CriticFinding.model_validate(entry))
        except Exception:
            continue
    notes = raw.get("notes")
    return CriticReport(
        critic=critic_name,
        score=score,
        on_target=on_target,
        findings=findings,
        notes=notes if isinstance(notes, str) else None,
    )


def merge_reports(
    heuristic: CriticReport,
    llm: CriticReport,
) -> CriticReport:
    """Merge a deterministic critic output with the LLM's. Lower score wins;
    findings are concatenated. Used by Voice + Emotion critics that have
    both a Python check and an LLM check."""
    score = min(heuristic.score, llm.score)
    on_target = heuristic.on_target and llm.on_target
    findings = list(heuristic.findings) + list(llm.findings)
    notes = "; ".join(s for s in (heuristic.notes, llm.notes) if s) or None
    return CriticReport(
        critic=heuristic.critic,
        score=round(score, 3),
        on_target=on_target,
        findings=findings,
        notes=notes,
    )


def _clamp_score(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, f))


# ---------------------------------------------------------------------------
# Dialogue extraction — Voice critic only cares about lines, not narration
# ---------------------------------------------------------------------------

import re

# Captures content between double quotes. Picks up most dialogue in
# straightforward prose; the LLM critic catches the rest.
_DIALOGUE_RE = re.compile(r"\"([^\"\n]{2,300})\"")


def extract_dialogue(prose: str) -> List[str]:
    return [m.group(1).strip() for m in _DIALOGUE_RE.finditer(prose or "")]


def severity_max(s1: Severity, s2: Severity) -> Severity:
    order = {"info": 0, "warn": 1, "error": 2}
    return s1 if order.get(s1, 0) >= order.get(s2, 0) else s2
