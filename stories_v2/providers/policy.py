"""Per-role provider preference lists.

The router walks the list in order on each call. If a provider is
unavailable (no API key) or returns a rate-limit / transient error, the
router rotates to the next entry.

Each entry is ``(provider_name, model_id, max_tokens, temperature_hint)``.
The temperature_hint is the default; callers may override per call.

**Together-first policy.** Every role tries Together AI first (the user's
primary provider — better creative writing on Kimi K2.5 and far more
generous per-minute limits than Groq's free tier). Groq remains as a
configured fallback for hard outages or Together rate limits.

Model picks within Together:
  - **moonshotai/Kimi-K2.5** owns creative roles (Architect, Chapter
    Planner, Scene Writer, Editor) — it produces the most natural,
    least AI-tell-laden prose in our stack.
  - **openai/gpt-oss-120b** owns structured-JSON workhorses (Profiler,
    World Builder, Character Forge, Continuity, Publisher) and the
    critic panel — it follows JSON schema instructions tightly and
    short critic prompts run quickly on Together.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional


class RouteSpec(NamedTuple):
    provider: str  # "groq" | "together"
    model_id: str
    max_tokens: int
    temperature: Optional[float] = None  # default, callers can override


# Model-ID shorthands so we have one place to flip them.
# Together (primary)
TOG_KIMI_K2_5 = "moonshotai/Kimi-K2.5"
TOG_GPT_OSS_120B = "openai/gpt-oss-120b"
TOG_EMBED = "intfloat/multilingual-e5-large-instruct"

# Groq (fallback only)
GROQ_LLAMA_70B = "llama-3.3-70b-versatile"
GROQ_KIMI_K2 = "moonshotai/kimi-k2-instruct-0905"
GROQ_GPT_OSS_120B = "openai/gpt-oss-120b"


DEFAULT_POLICY: Dict[str, List[RouteSpec]] = {
    # ---- Phase 1: context builders (long context, structured JSON) ----
    "profiler": [
        RouteSpec("together", TOG_GPT_OSS_120B, 4096, 0.6),
        RouteSpec("together", TOG_KIMI_K2_5, 4096, 0.6),
        RouteSpec("groq", GROQ_LLAMA_70B, 4096, 0.6),
    ],
    "world_builder": [
        RouteSpec("together", TOG_GPT_OSS_120B, 6000, 0.5),
        RouteSpec("together", TOG_KIMI_K2_5, 6000, 0.5),
        RouteSpec("groq", GROQ_LLAMA_70B, 6000, 0.5),
    ],
    "character_forge": [
        RouteSpec("together", TOG_GPT_OSS_120B, 6000, 0.6),
        RouteSpec("together", TOG_KIMI_K2_5, 6000, 0.6),
        RouteSpec("groq", GROQ_LLAMA_70B, 6000, 0.6),
    ],

    # ---- Phase 2: architecture (reasoning-heavy, structured JSON) ----
    "architect": [
        RouteSpec("together", TOG_KIMI_K2_5, 8000, 0.5),
        RouteSpec("together", TOG_GPT_OSS_120B, 8000, 0.5),
        RouteSpec("groq", GROQ_KIMI_K2, 8000, 0.5),
    ],
    "chapter_planner": [
        RouteSpec("together", TOG_KIMI_K2_5, 6000, 0.5),
        RouteSpec("together", TOG_GPT_OSS_120B, 6000, 0.5),
        RouteSpec("groq", GROQ_KIMI_K2, 6000, 0.5),
    ],

    # ---- Phase 3: drafting (best creative-prose model) ----
    "scene_writer": [
        RouteSpec("together", TOG_KIMI_K2_5, 4500, 0.85),
        RouteSpec("together", TOG_GPT_OSS_120B, 4500, 0.8),
        RouteSpec("groq", GROQ_KIMI_K2, 4500, 0.85),
    ],

    # ---- Phase 4: critic panel (parallel, low-cost, fast) ----
    # All five critics now target Together's gpt-oss-120b: it follows JSON
    # schemas tightly, has plenty of capacity for short critic prompts,
    # and avoids Groq's tight per-minute critic-fan-out budget.
    "critic_voice": [
        RouteSpec("together", TOG_GPT_OSS_120B, 1500, 0.2),
        RouteSpec("together", TOG_KIMI_K2_5, 1500, 0.2),
        RouteSpec("groq", GROQ_LLAMA_70B, 1500, 0.2),
    ],
    "critic_show_dont_tell": [
        RouteSpec("together", TOG_GPT_OSS_120B, 1500, 0.2),
        RouteSpec("together", TOG_KIMI_K2_5, 1500, 0.2),
        RouteSpec("groq", GROQ_LLAMA_70B, 1500, 0.2),
    ],
    "critic_ai_detect": [
        RouteSpec("together", TOG_GPT_OSS_120B, 1500, 0.2),
        RouteSpec("together", TOG_KIMI_K2_5, 1500, 0.2),
        RouteSpec("groq", GROQ_LLAMA_70B, 1500, 0.2),
    ],
    "critic_emotion": [
        RouteSpec("together", TOG_GPT_OSS_120B, 1500, 0.2),
        RouteSpec("together", TOG_KIMI_K2_5, 1500, 0.2),
        RouteSpec("groq", GROQ_LLAMA_70B, 1500, 0.2),
    ],
    "critic_pacing": [
        RouteSpec("together", TOG_GPT_OSS_120B, 1500, 0.2),
        RouteSpec("together", TOG_KIMI_K2_5, 1500, 0.2),
        RouteSpec("groq", GROQ_LLAMA_70B, 1500, 0.2),
    ],

    # ---- Phase 5: synthesis & polish ----
    "editor": [
        RouteSpec("together", TOG_KIMI_K2_5, 5000, 0.55),
        RouteSpec("together", TOG_GPT_OSS_120B, 5000, 0.55),
        RouteSpec("groq", GROQ_KIMI_K2, 5000, 0.55),
    ],
    "humaniser_llm": [
        RouteSpec("together", TOG_KIMI_K2_5, 3000, 0.7),
        RouteSpec("together", TOG_GPT_OSS_120B, 3000, 0.7),
        RouteSpec("groq", GROQ_LLAMA_70B, 3000, 0.7),
    ],

    # ---- Phase 6: memory + housekeeping (structured JSON workhorses) ----
    "continuity": [
        RouteSpec("together", TOG_GPT_OSS_120B, 6000, 0.3),
        RouteSpec("together", TOG_KIMI_K2_5, 6000, 0.3),
        RouteSpec("groq", GROQ_GPT_OSS_120B, 6000, 0.3),
    ],
    "publisher": [
        RouteSpec("together", TOG_GPT_OSS_120B, 4000, 0.4),
        RouteSpec("together", TOG_KIMI_K2_5, 4000, 0.4),
        RouteSpec("groq", GROQ_LLAMA_70B, 4000, 0.4),
    ],

    # ---- Embeddings ----
    "embed": [
        RouteSpec("together", TOG_EMBED, 0, None),
    ],
}
