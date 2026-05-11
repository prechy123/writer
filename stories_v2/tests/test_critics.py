"""Critic heuristic tests (no LLM calls).

Each critic exposes its deterministic component for direct invocation;
we exercise those paths so the structural / regex / statistical
detectors are verified without provider keys.
"""

from __future__ import annotations

import unittest

from stories_v2.agents.critics.ai_detect import _heuristic_ai_detect
from stories_v2.agents.critics.pacing import _heuristic_pacing
from stories_v2.agents.critics.show_dont_tell import _heuristic_show_dont_tell
from stories_v2.agents.critics.voice import _heuristic_voice_check
from stories_v2.schemas_v2 import (
    CharacterBibleV2,
    CharacterTier,
    LexicalFingerprint,
    VoiceFingerprint,
)


class AIDetectTests(unittest.TestCase):
    def test_em_dash_is_error(self):
        prose = "She paused — the door was open. She did not move."
        report = _heuristic_ai_detect(prose)
        errs = [f for f in report.findings if f.severity == "error"]
        self.assertTrue(any(f.field == "em_dash_present" for f in errs))
        self.assertLess(report.score, 1.0)

    def test_banned_phrase_flagged(self):
        prose = "She decided to delve into the tapestry of memory. The room was bustling."
        report = _heuristic_ai_detect(prose)
        fields = {f.field for f in report.findings}
        self.assertIn("banned_ai_phrase", fields)

    def test_low_burstiness_flagged(self):
        # 24 identical sentences -> stddev = 0, total_words > 200 -> should trigger
        prose = " ".join(
            ["This is a simple sentence that runs ten words long here."] * 24
        )
        report = _heuristic_ai_detect(prose)
        self.assertTrue(any(f.field == "low_burstiness" for f in report.findings),
                        f"got {report.findings}")

    def test_clean_human_prose_scores_high(self):
        prose = (
            "He stopped. The smell of burnt garlic. Mira's bowl was still on "
            "the counter, half full, and she would never have left it that way. "
            "He listened. The fridge hummed. Nothing else.\n\n"
            "Then a chair shifted upstairs."
        )
        report = _heuristic_ai_detect(prose)
        self.assertGreaterEqual(report.score, 0.9, f"findings={report.findings}")


class ShowDontTellTests(unittest.TestCase):
    def test_was_emotion_flagged(self):
        prose = "He was angry. He was very angry. She felt sad about it."
        report = _heuristic_show_dont_tell(prose)
        fields = {f.field for f in report.findings}
        self.assertIn("narrator_felt", fields)

    def test_wave_of_emotion_flagged(self):
        prose = "A wave of grief washed over her. She steadied herself on the rail."
        report = _heuristic_show_dont_tell(prose)
        fields = {f.field for f in report.findings}
        self.assertIn("wave_of_emotion", fields)

    def test_told_atmosphere_flagged(self):
        prose = "The room was tense. The air felt heavy."
        report = _heuristic_show_dont_tell(prose)
        fields = {f.field for f in report.findings}
        self.assertIn("told_atmosphere", fields)

    def test_clean_show_prose_scores_high(self):
        prose = (
            "He set the cup down too hard. The handle clinked against the saucer. "
            "Across the table, Mira didn't look up from the paper."
        )
        report = _heuristic_show_dont_tell(prose)
        self.assertGreaterEqual(report.score, 0.9, f"findings={report.findings}")


class PacingTests(unittest.TestCase):
    def test_weather_opener_flagged(self):
        prose = (
            "It was a cold and stormy night. He walked down the lane and considered his options. "
            * 5
        )
        report = _heuristic_pacing(prose, {"target_words": 600})
        fields = {f.field for f in report.findings}
        self.assertIn("opening_hook", fields)

    def test_short_scene_flagged_as_rushed(self):
        prose = "He arrived. He left."
        report = _heuristic_pacing(prose, {"target_words": 1000})
        fields = {f.field for f in report.findings}
        self.assertIn("length", fields)

    def test_empty_prose_is_error(self):
        report = _heuristic_pacing("", {"target_words": 600})
        self.assertEqual(report.score, 0.0)
        self.assertFalse(report.on_target)


class VoiceHeuristicTests(unittest.TestCase):
    def _char(self) -> CharacterBibleV2:
        return CharacterBibleV2(
            character_id="c1",
            story_id="s1",
            name="Liu Wei",
            tier=CharacterTier.MAIN,
            voice_fingerprint=VoiceFingerprint(
                lexical=LexicalFingerprint(
                    avg_sentence_words=8,
                    sentence_length_stddev=6,
                    contraction_rate=0.4,
                    formality=3,
                    style_register="street",
                ),
                banned_phrases=["delve"],
                sample_lines=["Try me. See what happens."],
            ),
        )

    def test_no_dialogue_returns_perfect(self):
        report = _heuristic_voice_check("Pure narration. No quotes.", [self._char()])
        self.assertEqual(report.score, 1.0)

    def test_banned_phrase_in_dialogue_flagged(self):
        prose = '"Let me delve into this," he said.'
        report = _heuristic_voice_check(prose, [self._char()])
        self.assertTrue(any(f.field == "banned_phrases" for f in report.findings),
                        f"got {report.findings}")
        self.assertLess(report.score, 0.85)


if __name__ == "__main__":
    unittest.main()
