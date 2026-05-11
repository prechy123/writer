"""Deterministic humanisation pipeline.

Runs AFTER the Editor's rewrite, BEFORE commit. Pure Python — no LLM.
Catches the slop the model still produces despite the system prelude.

Pipeline (in order):
    banned_tokens.strip      — em-dash → period+capital; phrase blocklist removed
    contractions.inject      — boost contractions in dialogue (don't, can't, ...)
    burstiness.enforce       — flag uniform sentence length (auto-split optional)
    fragments.flag           — flag absence of any sentence fragments
    idiom_inject.apply       — soft pass: ensure each main character has at least
                                one preferred-phrase mention if their dialogue
                                runs > a threshold
    detector_gate.check      — OPTIONAL: external AI-detector API gate

Public entry point: ``humanise(prose, *, present_characters=None, ...)``.
"""

from .pipeline import humanise

__all__ = ["humanise"]
