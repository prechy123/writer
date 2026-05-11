"""Voice fingerprint extraction, validation, and few-shot injection.

The schemas (``LexicalFingerprint``, ``VoiceFingerprint``) live in
``schemas_v2/voice.py`` because they're used by character bibles. This
package adds the *behaviour* — extracting fingerprints from sample text,
validating generated dialogue against a fingerprint, and building the
few-shot blocks the Scene Writer injects.

All public functions here are pure Python — no LLM calls, so they're
fast and unit-testable offline.
"""

from .extractor import extract_lexical_fingerprint, extract_voice_fingerprint
from .few_shot import build_few_shot_block, build_scene_few_shot
from .validator import score_dialogue_match, validate_voice

__all__ = [
    "extract_lexical_fingerprint",
    "extract_voice_fingerprint",
    "build_few_shot_block",
    "build_scene_few_shot",
    "score_dialogue_match",
    "validate_voice",
]
