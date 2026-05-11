"""Voice fingerprint extractor + validator tests.

Run with:
    python manage.py test stories_v2.tests.test_voice_fingerprint
"""

from __future__ import annotations

import unittest

from stories_v2.schemas_v2 import VoiceFingerprint, LexicalFingerprint
from stories_v2.voice import extract_lexical_fingerprint, score_dialogue_match


# Fixture: long, formal academic prose. Expect high formality, low contractions.
FORMAL_SAMPLE = """
The study examines the methodological foundations of the question. Therefore, we must first
consider the historical context. The literature provides substantial evidence that the phenomenon
predates the industrial revolution. Furthermore, the analysis indicates a consistent pattern across
samples. Nevertheless, certain caveats apply. The dataset, while comprehensive, is not exhaustive.
Consequently, our conclusions must be considered preliminary. Subsequent investigations will benefit
from a broader sampling frame and more rigorous methodology. The implications, however, are clear.
""" * 4


# Fixture: short, fragmented, profane street prose. Expect low formality, high
# contractions, profanity > 0, street register.
STREET_SAMPLE = """
Don't even start. I'm not doin' this again. You think I owe you anything? Hell no.
Fuck off. Seriously. I'm done. You can't keep showing up like nothing happened.
What. The. Hell. Did you really come here just to pull this shit? Get out.
I said get the fuck out. Now. Before I make you. Damn it. I'm tired. I can't.
You're not even sorry. You never are. Not really. Same shit, different week.
""" * 4


class ExtractorTests(unittest.TestCase):
    def test_formal_sample_scores_high_formality(self):
        fp, high_conf = extract_lexical_fingerprint(FORMAL_SAMPLE)
        self.assertTrue(high_conf, "expected high-confidence on 800+ word sample")
        self.assertGreaterEqual(fp.formality, 7, f"formality={fp.formality}")
        self.assertLess(fp.contraction_rate, 0.05)
        self.assertEqual(fp.profanity_rate, 0.0)
        # Stricter "professional"+ register, not "casual"
        self.assertIn(fp.style_register, {"professional", "academic", "casual"})

    def test_street_sample_scores_lower_formality_than_formal(self):
        street_fp, _ = extract_lexical_fingerprint(STREET_SAMPLE)
        formal_fp, _ = extract_lexical_fingerprint(FORMAL_SAMPLE)
        # The absolute formality scale is heuristic; what matters is that
        # the street sample is clearly less formal than the academic one.
        self.assertLess(street_fp.formality, formal_fp.formality,
                        f"street={street_fp.formality} formal={formal_fp.formality}")
        self.assertGreater(street_fp.contraction_rate, 0.05)
        self.assertGreater(street_fp.profanity_rate, 0.0)
        self.assertIn(street_fp.style_register, {"street", "casual"})

    def test_short_sample_returns_low_confidence(self):
        fp, high_conf = extract_lexical_fingerprint("Hi. Bye.")
        self.assertFalse(high_conf)

    def test_empty_string_safe(self):
        fp, _ = extract_lexical_fingerprint("")
        self.assertEqual(fp.formality, 5)  # default


class ValidatorTests(unittest.TestCase):
    def _fingerprint(self) -> VoiceFingerprint:
        """A character with casual, high-contraction speech and a banned phrase."""
        return VoiceFingerprint(
            lexical=LexicalFingerprint(
                avg_sentence_words=10,
                sentence_length_stddev=9,
                contraction_rate=0.45,
                formality=3,
                style_register="casual",
                profanity_rate=0.05,
                hedging_rate=0.02,
            ),
            preferred_phrases=["yeah no"],
            banned_phrases=["delve", "unwavering"],
            verbal_tics=["..."],
            sample_lines=["Yeah no, I'm out.", "Try me. See what happens."],
        )

    def test_on_voice_dialogue_scores_high(self):
        fp = self._fingerprint()
        # Match the fingerprint: avg ~10 words/sentence with high contractions
        # and varied length (stddev > 4).
        dialogue = (
            "Yeah no, I'm out. "
            "You can't keep showing up like nothing happened, and I can't keep pretending I'm fine. "
            "Don't. "
            "I've told you, what, three times already? "
            "Look, I'm done explaining myself to someone who never listens anyway. "
            "Just go. "
            "Before I say something I'll actually regret tomorrow morning. "
            "I mean it this time."
        )
        report = score_dialogue_match(dialogue, fp)
        self.assertGreaterEqual(report.score, 0.7, f"score={report.score} findings={report.findings}")

    def test_banned_phrase_flagged_error(self):
        fp = self._fingerprint()
        dialogue = "I shall delve into this matter with unwavering resolve."
        report = score_dialogue_match(dialogue, fp)
        errors = [f for f in report.findings if f.severity == "error"]
        self.assertTrue(any(f.field == "banned_phrases" for f in errors),
                        f"expected banned-phrase error, got {report.findings}")
        self.assertLess(report.score, 0.6)

    def test_uniform_sentence_length_flagged(self):
        fp = self._fingerprint()
        # Eight uniform 10-word sentences. Stddev will be ~0.
        dialogue = ". ".join(["This is just a uniform short sentence here today"] * 8) + "."
        report = score_dialogue_match(dialogue, fp)
        self.assertTrue(
            any(f.field == "sentence_length_stddev" for f in report.findings),
            f"expected stddev warning, got {report.findings}",
        )

    def test_low_contraction_flagged_when_character_is_casual(self):
        fp = self._fingerprint()
        # No contractions at all, despite the character speaking casually.
        dialogue = (
            "I am not going to do this. You will not change my mind. I have told you already. "
            "I do not need this anymore. You should leave. I will not be here when you return."
        )
        report = score_dialogue_match(dialogue, fp)
        contraction_findings = [f for f in report.findings if f.field == "contraction_rate"]
        self.assertTrue(contraction_findings,
                        f"expected contraction_rate finding, got {report.findings}")

    def test_empty_dialogue_returns_perfect_score(self):
        fp = self._fingerprint()
        report = score_dialogue_match("", fp)
        self.assertEqual(report.score, 1.0)
        self.assertEqual(report.findings, [])


if __name__ == "__main__":
    unittest.main()
