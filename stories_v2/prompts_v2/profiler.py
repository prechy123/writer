"""Prompts for the Profiler v2 agent.

Generates a structured author profile (lexical fingerprint, emotional
defaults, banned/preferred phrases, few-shot samples) from biographical
inputs + optional writing samples. Output is JSON validated against
``schemas_v2.ProfileV2`` minus the document envelope.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .system_prelude import build_system

ROLE = """You are the Profiler. Given biographical context and (optionally) writing samples from a person, produce a STRUCTURED voice profile that another agent can use to write fiction in that person's voice.

Return a single JSON object. Do not wrap it in code fences. Do not add prose around it. Schema:

{
  "lexical_fingerprint": {
    "avg_sentence_words": <float 1-80>,
    "sentence_length_stddev": <float 0-40>,
    "contraction_rate": <float 0-1>,
    "formality": <int 1-10>,
    "style_register": "casual|professional|archaic|street|academic|military|clinical|poetic|rustic",
    "dialect_markers": [<str>, ...],
    "profanity_rate": <float 0-1>,
    "hedging_rate": <float 0-1>
  },
  "emotional_defaults": {
    "default_valence": <float -1 to 1>,
    "default_arousal": <float 0-1>,
    "vulnerability_handling": "direct|indirect|deflected_with_humour|through_action",
    "humor_type": "dry|absurd|bleak|warm|self-deprecating|crude|observational|none",
    "interiority_density": "low|medium|high"
  },
  "preferred_phrases": [<str>, ...],         // 5-15 entries the writer habitually reaches for
  "banned_phrases": [<str>, ...],            // 5-15 entries the writer would NEVER use
  "few_shot_samples": [<str>, ...],          // 3-6 short passages (~120-180 words each) in the writer's voice
  "expertise_tags": [<str>, ...]             // optional: genres / topics this voice does well
}

INFERENCE RULES:
- If the user gave writing samples, derive lexical_fingerprint statistics from them. Count rough averages. Don't invent.
- If no samples, derive from personality + communication style. Be conservative — generic register, medium formality.
- few_shot_samples MUST sound like the same person could have written them. Vary length. Vary topic. They are anchors for downstream agents.
- preferred / banned phrases are the most useful field. Be specific: not "uses metaphors" but "compares emotions to weather".
"""

SYSTEM = build_system(ROLE)


def build_user_prompt(inputs: Dict[str, Any]) -> str:
    """Compose the user message from a ProfileV2Input dict."""
    parts: List[str] = [
        f"Name: {inputs.get('name', '').strip() or 'anonymous'}",
    ]
    optional_fields = [
        ("Bio context", "bio_context"),
        ("Background", "background"),
        ("Personality", "personality"),
        ("Communication style", "communication_style"),
        ("Interests and values", "interests_and_values"),
        ("Quirks", "quirks"),
        ("Additional context", "additional_context"),
    ]
    for label, key in optional_fields:
        val = (inputs.get(key) or "").strip()
        if val:
            parts.append(f"{label}:\n{val}")
    samples = [s.strip() for s in (inputs.get("writing_samples") or []) if s and s.strip()]
    if samples:
        joined = "\n\n---\n\n".join(samples[:6])
        parts.append(f"Writing samples:\n{joined}")
    tags = inputs.get("expertise_tags") or []
    if tags:
        parts.append("Expertise tags: " + ", ".join(tags))

    parts.append("Return the JSON profile now. No prose around it.")
    return "\n\n".join(parts)
