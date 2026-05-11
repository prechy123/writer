"""Deterministic humanisation pipeline tests."""

from __future__ import annotations

import asyncio
import unittest

from stories_v2.humanisation import humanise
from stories_v2.humanisation.banned_tokens import strip_blocklist, strip_em_dashes
from stories_v2.humanisation.burstiness import enforce as bursty_enforce
from stories_v2.humanisation.burstiness import measure
from stories_v2.humanisation.contractions import inject_in_dialogue
from stories_v2.humanisation.fragments import count_fragments


class EmDashTests(unittest.TestCase):
    def test_strip_interruption_dash(self):
        text = "She turned—slowly—and looked back."
        cleaned, count = strip_em_dashes(text)
        self.assertEqual(count, 2)
        self.assertNotIn("—", cleaned)

    def test_strip_zero_dashes_zero_count(self):
        text = "Plain prose. Nothing fancy."
        cleaned, count = strip_em_dashes(text)
        self.assertEqual(count, 0)
        self.assertEqual(cleaned, text)

    def test_post_strip_capitalises_next_word(self):
        text = "He paused—waited—moved on."
        cleaned, _ = strip_em_dashes(text)
        # Each splice produces "X. Y" with Y capitalised.
        self.assertIn(". W", cleaned)
        self.assertIn(". M", cleaned)


class BlocklistTests(unittest.TestCase):
    def test_delve_replaced(self):
        text = "She decided to delve into the mystery of the disappearance."
        cleaned, n = strip_blocklist(text)
        self.assertGreaterEqual(n, 1)
        self.assertNotIn("delve into", cleaned.lower())
        self.assertIn("looks into", cleaned.lower())

    def test_in_conclusion_stripped(self):
        text = "He thought about it. In conclusion, he refused."
        cleaned, n = strip_blocklist(text)
        self.assertGreaterEqual(n, 1)
        self.assertNotIn("in conclusion", cleaned.lower())

    def test_unwavering_to_steady(self):
        text = "His unwavering loyalty surprised her."
        cleaned, _ = strip_blocklist(text)
        self.assertIn("steady", cleaned.lower())
        self.assertNotIn("unwavering", cleaned.lower())


class ContractionTests(unittest.TestCase):
    def test_injects_in_dialogue_only(self):
        text = 'She said, "I am not going to do that." Then she walked away. I am not sure why.'
        cleaned, n = inject_in_dialogue(text)
        self.assertGreater(n, 0)
        # Inside quote: "I am not" -> "I'm not"  (or "isn't" path); narration UNCHANGED.
        self.assertIn('"', cleaned)
        self.assertIn("I am not sure why.", cleaned)  # narration preserved
        self.assertNotIn("I am not going", cleaned)   # dialogue contracted

    def test_no_dialogue_no_change(self):
        text = "I am not entirely sure what happened."
        cleaned, n = inject_in_dialogue(text)
        self.assertEqual(n, 0)
        self.assertEqual(cleaned, text)


class BurstinessTests(unittest.TestCase):
    def test_measure_uniform_low_stddev(self):
        text = " ".join(["This is a simple ten word sentence here right now."] * 6)
        _, stddev, count = measure(text)
        self.assertEqual(count, 6)
        self.assertLess(stddev, 1.0)

    def test_enforce_splits_long_compound(self):
        # All sentences identical length + each has a safe split point
        text = (
            "She walked into the dim corridor, and the lamps overhead were already on, and the door stood open. "
            * 8
        )
        cleaned, splits = bursty_enforce(text)
        self.assertGreater(splits, 0)
        # New stddev should be higher than original
        _, new_stddev, _ = measure(cleaned)
        _, old_stddev, _ = measure(text)
        self.assertGreater(new_stddev, old_stddev)

    def test_enforce_skips_when_already_bursty(self):
        text = "He stopped. The wind. A second of pure listening, and then the floorboard cracked under the next room's weight."
        cleaned, splits = bursty_enforce(text)
        self.assertEqual(splits, 0)


class FragmentTests(unittest.TestCase):
    def test_detects_fragments(self):
        text = "He stopped. The wind. He waited a long time. A whisper."
        frag, total = count_fragments(text)
        self.assertGreaterEqual(frag, 2)
        self.assertEqual(total, 4)

    def test_no_fragments_in_long_sentences(self):
        text = (
            "She walked into the dim corridor and discovered that the lamps overhead were already on. "
            "The door at the end of the hallway stood half-open, and the smell of cold metal drifted out."
        )
        frag, total = count_fragments(text)
        self.assertEqual(total, 2)
        self.assertLessEqual(frag, 0)


class PipelineTests(unittest.TestCase):
    def test_humanise_strips_em_dashes_and_phrases(self):
        text = (
            "She delved into the mystery — slowly, carefully, with unwavering focus. "
            "\"I am not afraid,\" she said. \"You will not stop me.\""
        )
        cleaned, report = asyncio.run(humanise(text))
        self.assertNotIn("—", cleaned)
        self.assertNotIn("delved into", cleaned.lower())
        self.assertNotIn("unwavering", cleaned.lower())
        self.assertGreaterEqual(report.em_dash_replacements, 1)
        self.assertGreaterEqual(report.banned_phrase_strikes, 2)
        self.assertGreaterEqual(report.contractions_injected, 1)

    def test_humanise_empty_safe(self):
        cleaned, report = asyncio.run(humanise(""))
        self.assertEqual(cleaned, "")
        self.assertEqual(report.em_dash_replacements, 0)


if __name__ == "__main__":
    unittest.main()
