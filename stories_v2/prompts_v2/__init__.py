"""Prompt templates for the v2 agents.

Every prompt in this package opens with ``system_prelude.PRELUDE`` —
the shared "write like a tired novelist, never like an LLM" instruction
that does the heavy lifting on humanisation at the model level. The
deterministic post-pass in ``humanisation/`` cleans up anything the
model still slips on.
"""

from .system_prelude import PRELUDE, build_system

__all__ = ["PRELUDE", "build_system"]
