"""Per-role provider preference lists — locked to Groq + Together AI.

The router walks the list in order on each call. If a provider is
unavailable (no API key) or returns a rate-limit / transient error, the
router rotates to the next entry.

Each entry is ``(provider_name, model_id, max_tokens, temperature_hint)``.
The temperature_hint is the default; callers may override per call.

Model picks reflect what each provider does best for the role:
  - Together: Kimi-K2.5 is the best free-tier creative-writing model in
    the stack, so it owns Scene Writer + Editor + Chapter Planner. The
    openai/gpt-oss-120b on Together is the best structured-output
    workhorse (good JSON adherence, long context), so it owns
    Continuity + Publisher.
  - Groq: Llama 3.3 70B is fast and follows instructions tightly, good
    for context-building (Profiler / World Builder / Character Forge)
    and for slow-moving heuristic critics. Llama 3.1 8B Instant is
    blazing fast and parallel-friendly, so the lighter critics
    (Voice / Show-Don't-Tell / AI-Detect) target it to keep the 5-way
    panel inside Groq's free-tier per-minute budget. Groq also hosts
    Kimi-K2 as a hot fallback when Together rate-limits the writer.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional


class RouteSpec(NamedTuple):
    provider: str  # "groq" | "together"
    model_id: str
    max_tokens: int
    temperature: Optional[float] = None  # default, callers can override


# Model-ID shorthands so we have one place to flip them.
# Groq
GROQ_LLAMA_70B = "llama-3.3-70b-versatile"
GROQ_LLAMA_8B = "llama-3.1-8b-instant"
GROQ_KIMI_K2 = "moonshotai/kimi-k2-instruct-0905"
GROQ_GPT_OSS_120B = "openai/gpt-oss-120b"

# Together
TOG_KIMI_K2_5 = "moonshotai/Kimi-K2.5"
TOG_GPT_OSS_120B = "openai/gpt-oss-120b"
TOG_EMBED = "intfloat/multilingual-e5-large-instruct"


DEFAULT_POLICY: Dict[str, List[RouteSpec]] = {
    # ---- Phase 1: context builders (long context, structured JSON) ----
    "profiler": [
        RouteSpec("groq", GROQ_LLAMA_70B, 4096, 0.6),
        RouteSpec("together", TOG_GPT_OSS_120B, 4096, 0.6),
    ],
    "world_builder": [
        RouteSpec("groq", GROQ_LLAMA_70B, 6000, 0.5),
        RouteSpec("together", TOG_GPT_OSS_120B, 6000, 0.5),
    ],
    "character_forge": [
        RouteSpec("groq", GROQ_LLAMA_70B, 6000, 0.6),
        RouteSpec("together", TOG_GPT_OSS_120B, 6000, 0.6),
    ],

    # ---- Phase 2: architecture (reasoning-heavy, structured JSON) ----
    "architect": [
        RouteSpec("together", TOG_KIMI_K2_5, 8000, 0.5),
        RouteSpec("groq", GROQ_KIMI_K2, 8000, 0.5),
        RouteSpec("groq", GROQ_LLAMA_70B, 8000, 0.5),
    ],
    "chapter_planner": [
        RouteSpec("together", TOG_KIMI_K2_5, 6000, 0.5),
        RouteSpec("groq", GROQ_KIMI_K2, 6000, 0.5),
        RouteSpec("groq", GROQ_LLAMA_70B, 6000, 0.5),
    ],

    # ---- Phase 3: drafting (best creative-prose model) ----
    "scene_writer": [
        RouteSpec("together", TOG_KIMI_K2_5, 4500, 0.85),
        RouteSpec("groq", GROQ_KIMI_K2, 4500, 0.85),
        RouteSpec("together", TOG_GPT_OSS_120B, 4500, 0.8),
    ],

    # ---- Phase 4: critic panel (parallel, low-cost, fast) ----
    # The three pattern-matcher critics use Llama 3.1 8B Instant on Groq
    # so the 5-way fan-out stays inside Groq's free-tier RPM budget and
    # finishes in under a second per critic.
    "critic_voice": [
        RouteSpec("groq", GROQ_LLAMA_8B, 1500, 0.2),
        RouteSpec("groq", GROQ_LLAMA_70B, 1500, 0.2),
        RouteSpec("together", TOG_GPT_OSS_120B, 1500, 0.2),
    ],
    "critic_show_dont_tell": [
        RouteSpec("groq", GROQ_LLAMA_8B, 1500, 0.2),
        RouteSpec("groq", GROQ_LLAMA_70B, 1500, 0.2),
        RouteSpec("together", TOG_GPT_OSS_120B, 1500, 0.2),
    ],
    "critic_ai_detect": [
        RouteSpec("groq", GROQ_LLAMA_8B, 1500, 0.2),
        RouteSpec("groq", GROQ_LLAMA_70B, 1500, 0.2),
        RouteSpec("together", TOG_GPT_OSS_120B, 1500, 0.2),
    ],
    # The two judgement-heavy critics need a stronger model.
    "critic_emotion": [
        RouteSpec("groq", GROQ_LLAMA_70B, 1500, 0.2),
        RouteSpec("together", TOG_GPT_OSS_120B, 1500, 0.2),
    ],
    "critic_pacing": [
        RouteSpec("groq", GROQ_LLAMA_70B, 1500, 0.2),
        RouteSpec("together", TOG_GPT_OSS_120B, 1500, 0.2),
    ],

    # ---- Phase 5: synthesis & polish ----
    "editor": [
        RouteSpec("together", TOG_KIMI_K2_5, 5000, 0.55),
        RouteSpec("groq", GROQ_KIMI_K2, 5000, 0.55),
        RouteSpec("groq", GROQ_LLAMA_70B, 5000, 0.6),
    ],
    "humaniser_llm": [
        RouteSpec("groq", GROQ_LLAMA_70B, 3000, 0.7),
        RouteSpec("together", TOG_GPT_OSS_120B, 3000, 0.7),
    ],

    # ---- Phase 6: memory + housekeeping (structured JSON workhorses) ----
    "continuity": [
        RouteSpec("together", TOG_GPT_OSS_120B, 6000, 0.3),
        RouteSpec("groq", GROQ_GPT_OSS_120B, 6000, 0.3),
        RouteSpec("groq", GROQ_LLAMA_70B, 6000, 0.3),
    ],
    "publisher": [
        RouteSpec("together", TOG_GPT_OSS_120B, 4000, 0.4),
        RouteSpec("groq", GROQ_LLAMA_70B, 4000, 0.4),
    ],

    # ---- Embeddings ----
    "embed": [
        RouteSpec("together", TOG_EMBED, 0, None),
    ],
}
