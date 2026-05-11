"""Corpus loader + retriever tests."""

from __future__ import annotations

import unittest

from stories_v2.corpus import list_exemplars, load_index, pick_exemplars


class CorpusLoaderTests(unittest.TestCase):
    def test_load_index_returns_entries(self):
        entries = load_index()
        self.assertGreater(len(entries), 0, "expected bundled exemplars to load")
        self.assertTrue(all(e.text for e in entries))
        # No em-dashes in our bundled exemplars (we wrote them ourselves).
        for e in entries:
            self.assertNotIn("—", e.text, f"em-dash in {e.id}")

    def test_list_exemplars_by_technique(self):
        results = list_exemplars(techniques=["sensory_anchoring"])
        self.assertGreater(len(results), 0)
        for e in results:
            self.assertIn("sensory_anchoring", [t.lower() for t in e.techniques])

    def test_pick_exemplars_ranks_by_overlap(self):
        results = pick_exemplars(
            techniques=["interruption", "contraction_dense_dialogue"],
            emotion_tags=["anger"],
            k=2,
        )
        self.assertGreaterEqual(len(results), 1)
        # Dialogue exemplars should rank above pure-action ones.
        self.assertTrue(
            any("interruption" in [t.lower() for t in e.techniques] for e in results),
            f"expected an interruption-tagged passage, got {[e.id for e in results]}",
        )

    def test_pick_exemplars_genre_filter(self):
        results = pick_exemplars(genres=["cultivation"], k=3)
        # cultivation OR generic should match
        for e in results:
            tags = {g.lower() for g in e.genres}
            self.assertTrue("cultivation" in tags or "generic" in tags,
                            f"unexpected genres: {e.genres}")


if __name__ == "__main__":
    unittest.main()
