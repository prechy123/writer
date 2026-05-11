"""Architect + Chapter Planner fallback path tests.

These don't exercise the LLM — they verify the deterministic shell that
runs when the provider chain is unreachable. The LLM happy path is
covered by Phase 11's end-to-end soak.
"""

from __future__ import annotations

import asyncio
import unittest

from stories_v2.agents.architect import _fallback_arc, find_act_for_chapter
from stories_v2.agents.chapter_planner import _fallback_chapter_plan
from stories_v2.emotion import chapter_reader_arc
from stories_v2.schemas_v2 import ArcPlan


class FallbackArcTests(unittest.TestCase):
    def test_act_split_covers_all_chapters_no_gaps(self):
        arc = _fallback_arc(story_id="s1", num_chapters=20, arc_preferences=None)
        # Acts must tile [0, 19] without gaps or overlap.
        covered = []
        for act in arc.acts:
            start, end = act.chapter_range
            covered.extend(range(start, end + 1))
        self.assertEqual(sorted(covered), list(range(20)))

    def test_short_story_still_produces_four_acts(self):
        arc = _fallback_arc(story_id="s1", num_chapters=4, arc_preferences=None)
        self.assertEqual(len(arc.acts), 4)
        # Even though acts may be 0-length-but-1-chapter wide, ranges must be valid.
        for act in arc.acts:
            self.assertLessEqual(act.chapter_range[0], act.chapter_range[1])

    def test_arc_preferences_are_honoured(self):
        prefs = {
            "cliffhanger_intensity": "low",
            "pacing_speed": "breakneck",
            "must_avoid_tropes": ["love_triangle"],
        }
        arc = _fallback_arc(story_id="s1", num_chapters=10, arc_preferences=prefs)
        self.assertEqual(arc.cliffhanger_intensity, "low")
        self.assertEqual(arc.pacing_speed, "breakneck")
        self.assertIn("love_triangle", arc.must_avoid_tropes)


class FindActTests(unittest.TestCase):
    def test_chapter_zero_is_in_discovery(self):
        arc = _fallback_arc(story_id="s1", num_chapters=20, arc_preferences=None)
        act_name, _ = find_act_for_chapter(arc, 0)
        self.assertEqual(act_name, "discovery")

    def test_final_chapter_is_in_catharsis(self):
        arc = _fallback_arc(story_id="s1", num_chapters=20, arc_preferences=None)
        act_name, pos = find_act_for_chapter(arc, 19)
        self.assertEqual(act_name, "catharsis")
        self.assertAlmostEqual(pos, (19 - arc.acts[-1].chapter_range[0]) /
                                    (arc.acts[-1].chapter_range[1] - arc.acts[-1].chapter_range[0] + 1),
                                    places=3)

    def test_position_in_act_is_zero_to_one(self):
        arc = _fallback_arc(story_id="s1", num_chapters=12, arc_preferences=None)
        for idx in range(12):
            _, pos = find_act_for_chapter(arc, idx)
            self.assertGreaterEqual(pos, 0.0)
            self.assertLess(pos, 1.0)


class FallbackChapterPlanTests(unittest.TestCase):
    def _arc(self) -> ArcPlan:
        return _fallback_arc(story_id="s1", num_chapters=10, arc_preferences=None)

    def test_four_scene_kishotenketsu(self):
        arc = self._arc()
        cast = [{"character_id": "c1", "name": "Liu Wei"}]
        reader_arc = chapter_reader_arc(act="discovery", chapter_position=0.0)
        plan = _fallback_chapter_plan(
            arc=arc,
            chapter_idx=0,
            target_chapter_words=2200,
            act_name="discovery",
            cast=cast,
            reader_arc=reader_arc,
        )
        self.assertEqual(len(plan.scenes), 4)
        phases = [s.kisho_phase for s in plan.scenes]
        self.assertEqual(phases, ["introduction", "development", "twist", "conclusion"])
        for sc in plan.scenes:
            self.assertEqual(sc.pov_character_id, "c1")
            self.assertGreater(sc.target_words, 200)

    def test_word_band_brackets_target(self):
        arc = self._arc()
        cast = [{"character_id": "c1", "name": "Liu Wei"}]
        reader_arc = chapter_reader_arc(act="discovery", chapter_position=0.0)
        plan = _fallback_chapter_plan(
            arc=arc,
            chapter_idx=0,
            target_chapter_words=2400,
            act_name="discovery",
            cast=cast,
            reader_arc=reader_arc,
        )
        self.assertLess(plan.target_word_floor, plan.target_chapter_words)
        self.assertGreater(plan.target_word_ceiling, plan.target_chapter_words)


if __name__ == "__main__":
    unittest.main()
