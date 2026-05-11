"""Emotion engine tests.

Run with:
    python manage.py test stories_v2.tests.test_emotion_engine
"""

from __future__ import annotations

import unittest

from stories_v2.emotion import (
    SceneEmotionTarget,
    apply_scene_event,
    build_scene_target,
    chapter_reader_arc,
    decay_toward_baseline,
    initial_mood,
    plutchik_add,
    plutchik_blend,
    plutchik_cosine_distance,
    plutchik_normalize,
    plutchik_scale,
    plutchik_to_valence_arousal,
    score_target_delivery,
    scene_reader_target,
)
from stories_v2.schemas_v2 import PlutchikVector


class PlutchikTests(unittest.TestCase):
    def test_add_clamps_at_one(self):
        a = PlutchikVector(joy=0.7)
        b = PlutchikVector(joy=0.8)
        c = plutchik_add(a, b)
        self.assertEqual(c.joy, 1.0)

    def test_scale_clamps_at_one(self):
        a = PlutchikVector(joy=0.5)
        c = plutchik_scale(a, 10)
        self.assertEqual(c.joy, 1.0)

    def test_blend_is_convex(self):
        a = PlutchikVector(joy=1.0)
        b = PlutchikVector(fear=1.0)
        c = plutchik_blend(a, b, weight_b=0.5)
        self.assertAlmostEqual(c.joy, 0.5, places=2)
        self.assertAlmostEqual(c.fear, 0.5, places=2)

    def test_cosine_distance_identical_vectors_zero(self):
        a = PlutchikVector(joy=0.5, fear=0.5)
        self.assertAlmostEqual(plutchik_cosine_distance(a, a), 0.0, places=4)

    def test_cosine_distance_opposite_vectors_one(self):
        a = PlutchikVector(joy=0.8)
        b = PlutchikVector(fear=0.8)
        # joy vs. fear are orthogonal axes — distance should be ~1.0
        self.assertAlmostEqual(plutchik_cosine_distance(a, b), 1.0, places=2)

    def test_zero_vector_distance_is_one(self):
        a = PlutchikVector()
        b = PlutchikVector(joy=0.5)
        self.assertEqual(plutchik_cosine_distance(a, b), 1.0)

    def test_normalize_preserves_direction(self):
        a = PlutchikVector(joy=0.3, fear=0.4)
        # Use a target_intensity that doesn't push any axis above the
        # 1.0 ceiling — otherwise clamping changes the ratio (by design).
        scaled = plutchik_normalize(a, target_intensity=0.25)
        ratio_a = a.joy / a.fear
        ratio_b = scaled.joy / scaled.fear
        self.assertAlmostEqual(ratio_a, ratio_b, places=3)

    def test_normalize_with_saturated_target_clamps(self):
        # When target_intensity is too high to satisfy without clamping,
        # the result saturates. This is expected behavior — documented in
        # the schema (axes are bounded [0, 1]).
        a = PlutchikVector(joy=0.3, fear=0.4)
        saturated = plutchik_normalize(a, target_intensity=0.9)
        self.assertLessEqual(saturated.joy, 1.0)
        self.assertLessEqual(saturated.fear, 1.0)


class ValenceArousalTests(unittest.TestCase):
    def test_joy_is_positive_valence(self):
        va = plutchik_to_valence_arousal(PlutchikVector(joy=0.9))
        self.assertGreater(va.valence, 0)
        self.assertGreater(va.arousal, 0.4)

    def test_fear_is_negative_valence_high_arousal(self):
        va = plutchik_to_valence_arousal(PlutchikVector(fear=0.9))
        self.assertLess(va.valence, 0)
        self.assertGreater(va.arousal, 0.7)

    def test_sadness_is_low_arousal(self):
        va_sad = plutchik_to_valence_arousal(PlutchikVector(sadness=0.9))
        va_fear = plutchik_to_valence_arousal(PlutchikVector(fear=0.9))
        self.assertLess(va_sad.arousal, va_fear.arousal)

    def test_va_within_range(self):
        for sample in [PlutchikVector(joy=1.0, fear=1.0, anger=1.0), PlutchikVector()]:
            va = plutchik_to_valence_arousal(sample)
            self.assertGreaterEqual(va.valence, -1.0)
            self.assertLessEqual(va.valence, 1.0)
            self.assertGreaterEqual(va.arousal, 0.0)
            self.assertLessEqual(va.arousal, 1.0)


class MoodStateTests(unittest.TestCase):
    def test_initial_mood_neutral(self):
        snap = initial_mood()
        self.assertEqual(snap.chapter_idx, 0)
        self.assertEqual(snap.scene_idx, 0)
        # Trust + anticipation seeded; nothing else heavy
        self.assertGreater(snap.plutchik.trust, 0)
        self.assertEqual(snap.plutchik.anger, 0)

    def test_apply_scene_event_increases_target_axis(self):
        snap = initial_mood()
        new = apply_scene_event(
            snap,
            delta={"anger": 0.5, "fear": 0.2},
            chapter_idx=0,
            scene_idx=1,
            event_summary="betrayed by mentor",
            apply_decay=False,
        )
        self.assertGreater(new.plutchik.anger, snap.plutchik.anger)
        self.assertGreater(new.plutchik.fear, snap.plutchik.fear)
        self.assertEqual(new.scene_idx, 1)
        self.assertEqual(new.last_event_summary, "betrayed by mentor")

    def test_decay_pulls_toward_baseline(self):
        spiked = PlutchikVector(anger=0.9, joy=0.0)
        decayed = decay_toward_baseline(spiked)
        # Anger should drop, joy should rise slightly (toward baseline)
        self.assertLess(decayed.anger, spiked.anger)
        # Baseline trust 0.3 — anger has no trust contribution; just confirm clamp
        self.assertGreaterEqual(decayed.anger, 0.0)
        self.assertLessEqual(decayed.anger, 1.0)

    def test_apply_scene_event_with_decay(self):
        snap = initial_mood()
        new = apply_scene_event(
            snap,
            delta={"anger": 0.6},
            chapter_idx=0,
            scene_idx=1,
            apply_decay=True,
        )
        # With decay, anger should be slightly less than 0.6 raw.
        self.assertLess(new.plutchik.anger, 0.6)
        self.assertGreater(new.plutchik.anger, 0.2)


class ReaderArcTests(unittest.TestCase):
    def test_chapter_arc_returns_four_phases(self):
        arc = chapter_reader_arc(act="escalation", chapter_position=0.5)
        self.assertEqual(set(arc.keys()), {"introduction", "development", "twist", "conclusion"})

    def test_cliffhanger_boosts_conclusion(self):
        low = chapter_reader_arc(act="escalation", chapter_position=0.5, cliffhanger_intensity="low")
        high = chapter_reader_arc(act="escalation", chapter_position=0.5, cliffhanger_intensity="high")
        from stories_v2.emotion.plutchik import intensity as plut_intensity
        self.assertGreater(plut_intensity(high["conclusion"]), plut_intensity(low["conclusion"]))

    def test_scene_target_maps_first_to_intro(self):
        arc = chapter_reader_arc(act="discovery", chapter_position=0.0)
        target = scene_reader_target(arc, scene_idx=0, total_scenes=4)
        # First scene of a 4-scene chapter is introduction
        self.assertEqual(target.model_dump(), arc["introduction"].model_dump())

    def test_scene_target_maps_last_to_conclusion(self):
        arc = chapter_reader_arc(act="escalation", chapter_position=0.7)
        target = scene_reader_target(arc, scene_idx=3, total_scenes=4)
        self.assertEqual(target.model_dump(), arc["conclusion"].model_dump())


class SceneTargetTests(unittest.TestCase):
    def test_on_target_when_delivered_close(self):
        target = build_scene_target(
            pov_character_id="c1",
            pov_character_name="Liu Wei",
            protagonist_current_mood=PlutchikVector(anticipation=0.6, fear=0.2),
            target_reader_vector=PlutchikVector(fear=0.7, anticipation=0.4),
        )
        report = score_target_delivery(
            target,
            delivered_protagonist_end=target.protagonist_end,
            delivered_reader_end=target.reader_end,
        )
        self.assertTrue(report.on_target)
        self.assertEqual(report.severity, "info")

    def test_drift_when_partial_miss(self):
        target = build_scene_target(
            pov_character_id="c1",
            pov_character_name="Liu Wei",
            protagonist_current_mood=PlutchikVector(joy=0.7),
            target_reader_vector=PlutchikVector(fear=0.8),
        )
        # Deliver a partial mismatch.
        report = score_target_delivery(
            target,
            delivered_protagonist_end=PlutchikVector(joy=0.4, fear=0.3),
            delivered_reader_end=PlutchikVector(fear=0.3, sadness=0.5),
        )
        self.assertIn(report.severity, {"warn", "error"})

    def test_total_miss_flagged_error(self):
        target = build_scene_target(
            pov_character_id="c1",
            pov_character_name="Liu Wei",
            protagonist_current_mood=PlutchikVector(joy=0.7),
            target_reader_vector=PlutchikVector(fear=0.8),
        )
        # Deliver pure joy where fear was wanted.
        report = score_target_delivery(
            target,
            delivered_protagonist_end=PlutchikVector(joy=0.9),
            delivered_reader_end=PlutchikVector(joy=0.9),
        )
        self.assertEqual(report.severity, "error")
        self.assertFalse(report.on_target)


if __name__ == "__main__":
    unittest.main()
