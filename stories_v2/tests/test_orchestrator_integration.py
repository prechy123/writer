"""End-to-end orchestrator integration test (mocked router).

Runs the full v2 pipeline:
    Quick survey → bibles → start → arc → chapter loop → scenes commit

The provider router is monkeypatched to return deterministic, contract-
compliant responses for each role. This proves the orchestrator wiring
is correct without making real API calls.

The user can swap the mock for a real router by setting any of the
provider API keys in .env and removing the monkeypatch.
"""

from __future__ import annotations

import asyncio
import json
import time
import unittest
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import patch

from stories_v2 import mongo as mg
from stories_v2.engine.orchestrator import run_story_async
from stories_v2.schemas_v2 import (
    CharacterTier,
    StoryStatus,
)


# ---------------------------------------------------------------------------
# Deterministic mock router
# ---------------------------------------------------------------------------

class MockRouter:
    """A router that returns deterministic, schema-shaped responses per role.

    Used in tests to exercise the orchestrator path end-to-end without
    making any HTTP calls. The mock prose deliberately uses contractions,
    varied sentence lengths, NO em-dashes, and no banned phrases — so
    the critic panel finds little to complain about and the editor
    rarely needs to rewrite.
    """

    def available_providers(self) -> List[str]:
        return ["mock"]

    async def chat_text(
        self,
        *,
        role: str,
        system: Optional[str],
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 120.0,
    ) -> str:
        if role == "scene_writer":
            return _scene_prose()
        if role == "editor":
            # Editor is asked to rewrite — return the input prose unchanged.
            for msg in reversed(messages):
                if "=== CURRENT DRAFT ===" in (msg.get("content") or ""):
                    parts = msg["content"].split("=== CURRENT DRAFT ===")
                    return parts[1].split("=== ")[0].strip()
            return _scene_prose()
        return "ok"

    async def chat_json(
        self,
        *,
        role: str,
        system: Optional[str],
        messages: List[Dict[str, str]],
        schema: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 180.0,
    ) -> Dict[str, Any]:
        if role == "profiler":
            return {
                "lexical_fingerprint": {"avg_sentence_words": 10, "sentence_length_stddev": 6, "contraction_rate": 0.45, "formality": 4, "style_register": "casual", "dialect_markers": [], "profanity_rate": 0.0, "hedging_rate": 0.1},
                "emotional_defaults": {"default_valence": 0.1, "default_arousal": 0.5, "vulnerability_handling": "indirect", "humor_type": "dry", "interiority_density": "medium"},
                "preferred_phrases": [], "banned_phrases": [], "few_shot_samples": [], "expertise_tags": [],
            }
        if role == "world_builder":
            return {
                "setting": "a small mountain town in the late nineteenth century",
                "time_period": "1890s", "technology_level": "early industrial",
                "social_structure": "frontier with rigid class stratification",
                "geography": "high alpine valley, one road in, one road out",
                "languages": ["English"],
                "factions": [],
                "magic_or_system": {"kind": "none", "description": "", "progression_path": [], "cost_or_drawback": "", "hard_limits": []},
                "rules": [{"scope": "global", "rule": "winter cuts the town off for three months", "consequence_if_broken": ""}],
                "banned_anachronisms": ["smartphone", "automobile", "antibiotics"],
                "must_have_vibes": ["coal smoke", "frozen pump handles", "lantern light"],
            }
        if role == "character_forge":
            return {"characters": [
                {
                    "name": "Mara Holt", "tier": "main", "role": "protagonist",
                    "short_description": "the town's only doctor, exhausted and stubborn",
                    "portrait_blurb": "Forty, lean, smells of camphor.",
                    "background": "Trained in Boston, returned home after her father's death.",
                    "motivations": ["keep everyone alive through winter"],
                    "fears": ["losing another patient she could have saved"],
                    "arc": "Learns to ask for help.",
                    "voice_fingerprint": {
                        "lexical": {"avg_sentence_words": 9, "sentence_length_stddev": 5, "contraction_rate": 0.5, "formality": 3, "style_register": "casual", "dialect_markers": [], "profanity_rate": 0.02, "hedging_rate": 0.05},
                        "preferred_phrases": ["I'd say", "look"],
                        "banned_phrases": ["delve"],
                        "verbal_tics": [], "catchphrases": [],
                        "sample_lines": ["I'd say you've got about an hour.", "Look. Sit down. Drink that."],
                        "silence_style": "long pause then a flat statement",
                    },
                },
                {
                    "name": "Sam Pell", "tier": "recurring", "role": "stableman",
                    "short_description": "loyal, broke, knows everyone's secrets",
                    "portrait_blurb": "Twenty-four, all elbows.",
                    "voice_fingerprint": {
                        "lexical": {"avg_sentence_words": 6, "sentence_length_stddev": 4, "contraction_rate": 0.55, "formality": 2, "style_register": "casual", "dialect_markers": [], "profanity_rate": 0.05, "hedging_rate": 0.0},
                        "preferred_phrases": ["yeah"], "banned_phrases": [],
                        "verbal_tics": [], "catchphrases": [],
                        "sample_lines": ["Yeah no, that ain't right.", "Want me to walk you home?"],
                    },
                },
            ]}
        if role == "architect":
            return {
                "arc_name": "The Long Winter",
                "arc_theme": "isolation forces honesty",
                "target_reader_journey": "From quiet dread to earned warmth.",
                "acts": [
                    {"name": "discovery", "chapter_range": [0, 0], "promise": "Set the town and the doctor.", "key_beats": ["arrival of the storm"]},
                    {"name": "escalation", "chapter_range": [1, 1], "promise": "Pressure mounts.", "key_beats": ["a patient she can't save alone"]},
                    {"name": "revelation", "chapter_range": [2, 2], "promise": "What she's been hiding.", "key_beats": []},
                    {"name": "catharsis", "chapter_range": [3, 3], "promise": "She asks for help.", "key_beats": []},
                ],
                "progression_milestones": [],
                "plot_seeds": [],
                "subplots": [],
                "must_include_tropes": [], "must_avoid_tropes": [],
                "cliffhanger_intensity": "medium", "pacing_speed": "slow_burn",
                "romance_temperature": "subtext", "action_density": "light",
            }
        if role == "chapter_planner":
            return _chapter_plan()
        if role == "continuity":
            return {
                "scene_summary": "Mara delivers a difficult diagnosis. The storm intensifies.",
                "key_dialogue": ["I'd say you've got about an hour."],
                "character_mood_deltas": [
                    {"character_id": "REPLACED_AT_RUNTIME", "name": "Mara Holt",
                     "plutchik_delta": {"sadness": 0.1, "anticipation": 0.05},
                     "last_event_summary": "told a patient he was dying"},
                ],
                "protagonist_emotion_end": {"joy": 0.0, "trust": 0.2, "fear": 0.3, "surprise": 0.0, "sadness": 0.45, "disgust": 0.0, "anger": 0.05, "anticipation": 0.4},
                "reader_emotion_end": {"joy": 0.0, "trust": 0.2, "fear": 0.4, "surprise": 0.1, "sadness": 0.5, "disgust": 0.0, "anger": 0.0, "anticipation": 0.55},
                "world_state_changes": [],
                "plot_seed_events": [],
                "open_threads_added": ["mara hasn't slept in two days"],
                "open_threads_closed": [],
                "unresolved_cliffhangers": [],
            }
        # Critic roles
        if role.startswith("critic_"):
            return {"score": 0.92, "on_target": True, "findings": []}
        return {}

    async def chat_stream(self, **kwargs) -> AsyncIterator[str]:  # pragma: no cover
        yield "mock"

    async def embed(self, text: str, *, role: str = "embed") -> Optional[List[float]]:
        # Deterministic 8-dim "embedding" so the in-mem cosine still works
        if not text:
            return None
        h = abs(hash(text))
        return [((h >> (i * 3)) & 0xFF) / 255.0 for i in range(8)]


def _scene_prose() -> str:
    """A short, contraction-rich, fragment-friendly scene with no em-dashes."""
    return (
        "Mara hung her coat by the door. It dripped. The lamp on the hall table "
        "was nearly out, and she did not bother to fix it.\n\n"
        "\"You're back,\" Sam said from the kitchen.\n\n"
        "\"I'm back.\"\n\n"
        "He didn't ask. He poured her a cup of something dark, and she sat down "
        "across from him, and the snow kept coming down outside, and for a long "
        "while neither of them said anything else. The clock ticked. The fire "
        "moved. She set the cup down too hard on the second sip and looked at "
        "the back of her own hand like she didn't recognise it.\n\n"
        "\"Henry,\" she said finally. \"Tomorrow at the earliest.\"\n\n"
        "Sam nodded. He didn't say anything for a long time after that, and she "
        "was grateful for it."
    )


def _chapter_plan() -> Dict[str, Any]:
    """A 2-scene Kishōtenketsu plan referencing valid character_ids."""
    return {
        "chapter_idx": 0,
        "chapter_number": 1,
        "chapter_title": "The Long Walk Home",
        "chapter_summary": "Mara comes back to town in the first storm.",
        "act_name": "discovery",
        "chapter_position_in_act": 0.0,
        "opening_hook": "A coat that drips on the floor.",
        "cliffhanger": "Henry's chances by morning.",
        "progression_reward": "",
        "scenes": [
            {
                "scene_idx": 0,
                "title": "Arrival",
                "summary": "Mara returns home in the storm.",
                "kisho_phase": "introduction",
                "pov_character_id": "REPLACED_AT_RUNTIME_0",
                "pov_character_name": "Mara Holt",
                "present_character_ids": [],
                "location": "Mara's hallway",
                "time_of_day": "late evening",
                "goal": "get warm and eat",
                "conflict": "exhaustion + bad news to deliver",
                "disaster": "she sets the cup down too hard",
                "protagonist_start_emotion": {"sadness": 0.3, "fear": 0.2, "trust": 0.2, "joy": 0.0, "surprise": 0.0, "disgust": 0.0, "anger": 0.0, "anticipation": 0.3},
                "protagonist_end_emotion":   {"sadness": 0.5, "fear": 0.3, "trust": 0.2, "joy": 0.0, "surprise": 0.0, "disgust": 0.0, "anger": 0.05, "anticipation": 0.4},
                "reader_start_emotion":      {"trust": 0.3, "anticipation": 0.4, "joy": 0.0, "fear": 0.2, "sadness": 0.1, "surprise": 0.0, "disgust": 0.0, "anger": 0.0},
                "reader_end_emotion":        {"trust": 0.3, "anticipation": 0.5, "fear": 0.4, "sadness": 0.4, "joy": 0.0, "surprise": 0.1, "disgust": 0.0, "anger": 0.0},
                "sensory_focus": ["touch", "sound"],
                "interiority_density": "medium",
                "techniques": ["sensory_anchoring", "subtext", "contraction_dense_dialogue"],
                "target_words": 350,
            },
            {
                "scene_idx": 1,
                "title": "The Telling",
                "summary": "She finally says it.",
                "kisho_phase": "conclusion",
                "pov_character_id": "REPLACED_AT_RUNTIME_0",
                "pov_character_name": "Mara Holt",
                "present_character_ids": [],
                "location": "Mara's kitchen",
                "time_of_day": "later that night",
                "goal": "name the prognosis out loud",
                "conflict": "she does not want to",
                "disaster": "she says it",
                "protagonist_start_emotion": {"sadness": 0.5, "anticipation": 0.4, "fear": 0.3, "joy": 0.0, "surprise": 0.0, "trust": 0.2, "disgust": 0.0, "anger": 0.05},
                "protagonist_end_emotion":   {"sadness": 0.55, "trust": 0.3, "fear": 0.2, "anticipation": 0.3, "joy": 0.0, "surprise": 0.0, "disgust": 0.0, "anger": 0.0},
                "reader_start_emotion":      {"anticipation": 0.55, "fear": 0.4, "sadness": 0.4, "joy": 0.0, "trust": 0.2, "surprise": 0.0, "disgust": 0.0, "anger": 0.0},
                "reader_end_emotion":        {"sadness": 0.55, "trust": 0.3, "anticipation": 0.3, "fear": 0.2, "joy": 0.0, "surprise": 0.0, "disgust": 0.0, "anger": 0.0},
                "sensory_focus": ["sound", "touch"],
                "interiority_density": "high",
                "techniques": ["silence_and_deflection", "subtext"],
                "target_words": 300,
            },
        ],
        "target_chapter_words": 700,
        "target_word_floor": 500,
        "target_word_ceiling": 1000,
    }


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------

class _Cleanup:
    """Helper to make sure we tear down test docs even on failure."""
    def __init__(self):
        self.story_id = f"test-{uuid.uuid4()}"

    def remove(self):
        for col_name in [mg.COL_STORIES, mg.COL_CHARACTERS, mg.COL_WORLDS, mg.COL_BEATS, mg.COL_SCENES, mg.COL_EVENTS]:
            try:
                mg.col(col_name).delete_many({
                    "$or": [
                        {"_id": self.story_id},
                        {"story_id": self.story_id},
                    ]
                })
            except Exception:
                pass


def _seed_story(story_id: str) -> None:
    """Pre-create a story envelope + cast + world for the orchestrator to find.

    Skipping the wizard endpoints keeps this test independent of provider
    keys; we exercise the orchestrator's read path directly.
    """
    char_main_id = f"{story_id}-char-mara"
    char_rec_id = f"{story_id}-char-sam"

    mg.col(mg.COL_STORIES).insert_one({
        "_id": story_id,
        "title": "The Long Winter",
        "status": StoryStatus.PENDING.value,
        "quick_survey": {
            "title": "The Long Winter",
            "premise": "A frontier doctor faces an impossible winter.",
            "num_chapters": 1,
            "initial_chapters": 1,
            "genres": ["literary"],
            "tone": ["gritty", "hopeful"],
            "characters": [],
            "pov": "third_limited",
            "tense": "past",
            "target_chapter_words": 700,
        },
        "deep_survey": None,
        "arc_plan": None,
        "chapters": [],
        "current_chapter_idx": 0,
        "progress": {"stage": "queued", "percent": 0},
        "progress_log": [],
        "created_at": None,
        "updated_at": None,
        "hidden": False,
        "character_budget": {"num_chapters": 1, "main": [1, 1], "recurring": [1, 2], "side": [0, 2]},
    })

    mg.col(mg.COL_CHARACTERS).insert_many([
        {
            "_id": f"{story_id}:{char_main_id}",
            "story_id": story_id,
            "character_id": char_main_id,
            "name": "Mara Holt",
            "tier": "main",
            "role": "protagonist",
            "short_description": "the doctor",
            "portrait_blurb": "forty, lean",
            "voice_fingerprint": {
                "lexical": {"avg_sentence_words": 9, "sentence_length_stddev": 5, "contraction_rate": 0.5, "formality": 3, "style_register": "casual", "dialect_markers": [], "profanity_rate": 0.02, "hedging_rate": 0.05},
                "preferred_phrases": [], "banned_phrases": [], "verbal_tics": [], "catchphrases": [],
                "sample_lines": ["I'd say you've got about an hour."],
                "silence_style": None,
            },
            "mood_state_history": [],
        },
        {
            "_id": f"{story_id}:{char_rec_id}",
            "story_id": story_id,
            "character_id": char_rec_id,
            "name": "Sam Pell",
            "tier": "recurring",
            "role": "stableman",
            "short_description": "loyal, broke",
            "portrait_blurb": "twenty-four",
            "voice_fingerprint": None,
            "mood_state_history": [],
        },
    ])

    mg.col(mg.COL_WORLDS).insert_one({
        "story_id": story_id,
        "setting": "frontier town",
        "rules": [{"scope": "global", "rule": "winter cuts the town off"}],
        "banned_anachronisms": ["smartphone"],
        "must_have_vibes": ["coal smoke"],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class OrchestratorIntegrationTests(unittest.TestCase):
    """Run the full orchestrator with a mocked router. ~1-3s on a laptop."""

    def setUp(self):
        self.cu = _Cleanup()
        _seed_story(self.cu.story_id)

    def tearDown(self):
        self.cu.remove()

    def test_one_chapter_two_scenes_full_pipeline(self):
        story_id = self.cu.story_id
        mock = MockRouter()

        # Patch the global router so every agent picks up the mock.
        with patch("stories_v2.providers.router._router_instance", mock), \
             patch("stories_v2.providers.router.get_router", return_value=mock), \
             patch("stories_v2.providers.get_router", return_value=mock):
            # Patch the per-module router lookups too — agents import get_router
            # from the package level, so the package-level patch covers them.
            t0 = time.time()
            asyncio.run(run_story_async(story_id, batch_size=1))
            elapsed = time.time() - t0

        # Allow up to 30s on slow CI / mongo — the mock itself is ~instant.
        self.assertLess(elapsed, 30.0, f"orchestrator too slow: {elapsed:.2f}s")

        # Story envelope should be completed
        story = mg.get_story_envelope(story_id)
        self.assertIsNotNone(story)
        self.assertEqual(story["status"], StoryStatus.COMPLETED.value)
        self.assertEqual(story.get("current_chapter_idx"), 1)
        chapters = story.get("chapters") or []
        self.assertEqual(len(chapters), 1)
        self.assertGreater(chapters[0].get("word_count", 0), 50)
        self.assertNotIn("—", chapters[0].get("text", ""), "em-dash leaked into final prose")

        # Two scene drafts should be committed
        scenes = list(mg.col(mg.COL_SCENES).find({"story_id": story_id}))
        self.assertEqual(len(scenes), 2)
        for sc in scenes:
            self.assertEqual(sc["status"], "committed")
            self.assertNotIn("—", sc["final_prose"])
            self.assertIsNotNone(sc.get("summary"))
            self.assertIsNotNone(sc.get("humanisation_report"))

        # Beat sheet should be persisted
        beat = mg.col(mg.COL_BEATS).find_one({"story_id": story_id, "chapter_idx": 0})
        self.assertIsNotNone(beat)
        self.assertEqual(len(beat.get("scenes") or []), 2)

        # Run events should include the expected milestones
        events = list(mg.col(mg.COL_EVENTS).find({"story_id": story_id}).sort("seq", 1))
        event_types = {e["event_type"] for e in events}
        for expected in {"story.queued", "arc.planned", "chapter.planning", "chapter.planned",
                         "scene.started", "scene.drafted", "scene.committed",
                         "chapter.committed", "story.completed"}:
            self.assertIn(expected, event_types, f"missing event type: {expected}")


if __name__ == "__main__":
    unittest.main()
