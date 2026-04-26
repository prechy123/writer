from django.test import SimpleTestCase, override_settings

from .agents import _get_llm, _max_tokens_for_role, _model_for_role
from .prompts import (
    continuity_extractor_prompt,
    launch_chapter_planner_prompt,
    perfectionist_prompt,
    reviewer_prompt,
    storyteller_prompt,
    writer_prompt,
)
from .schemas import CharacterState, LaunchChapterPlan, StoryPlanSchema


class TogetherModelRoutingTests(SimpleTestCase):
    @override_settings(
        TOGETHER_MODELS={
            "default": "MiniMaxAI/MiniMax-M2.7",
            "writer": "moonshotai/Kimi-K2.5",
            "reviewer": "openai/gpt-oss-120b",
        }
    )
    def test_model_for_role_uses_specific_serverless_model(self):
        self.assertEqual(_model_for_role("writer"), "moonshotai/Kimi-K2.5")
        self.assertEqual(_model_for_role("reviewer"), "openai/gpt-oss-120b")

    @override_settings(
        TOGETHER_MODELS={"default": "MiniMaxAI/MiniMax-M2.7"},
        TOGETHER_DEFAULT_MODEL="MiniMaxAI/MiniMax-M2.7",
        TOGETHER_MODEL="legacy-model",
    )
    def test_model_for_role_falls_back_to_default(self):
        self.assertEqual(_model_for_role("unknown"), "MiniMaxAI/MiniMax-M2.7")

    @override_settings(
        TOGETHER_MAX_TOKENS={"default": 1234, "storyteller": 9876},
        TOGETHER_DEFAULT_MAX_TOKENS=1234,
    )
    def test_max_tokens_for_role_uses_specific_budget(self):
        self.assertEqual(_max_tokens_for_role("storyteller"), 9876)
        self.assertEqual(_max_tokens_for_role("unknown"), 1234)

    @override_settings(
        TOGETHER_MAX_TOKENS={"default": 1234, "storyteller": 0},
        TOGETHER_DEFAULT_MAX_TOKENS=1234,
    )
    def test_max_tokens_for_role_allows_disabling_explicit_budget(self):
        self.assertIsNone(_max_tokens_for_role("storyteller"))

    @override_settings(
        TOGETHER_API_KEY="test-key",
        TOGETHER_MODELS={"default": "MiniMaxAI/MiniMax-M2.7"},
        TOGETHER_MAX_TOKENS={"default": 4321},
        TOGETHER_DEFAULT_MAX_TOKENS=4321,
    )
    def test_get_llm_applies_output_budget(self):
        llm = _get_llm(role="default")
        self.assertEqual(llm.max_tokens, 4321)


class StoryQualitySchemaTests(SimpleTestCase):
    def test_story_plan_accepts_voice_and_background_fields(self):
        plan = StoryPlanSchema(
            title="Dust and Signal",
            genre="Coming-of-age",
            webnovel_strategy={
                "platform_genre": "Urban",
                "lead_type": "male lead",
                "primary_tags": ["WEAKTOSTRONG"],
                "reader_promise": "A poor boy earns visible wins through discipline.",
                "protagonist_cheat": "Exceptional memory for overheard lessons.",
            },
            launch_chapter_plan={
                "conversion_goal": "Make readers root for Tare and add the book.",
                "first_200_words_hook": "Tare opens a rival-marked book and finds a deadline.",
                "chapter_one_progression_reward": "Tare discovers the scholarship clue.",
                "chapter_one_cliffhanger": "The book's owner arrives at his door.",
            },
            setting="A low-income Lagos neighborhood",
            themes=["ambition"],
            characters=[
                {
                    "name": "Tare",
                    "role": "protagonist",
                    "description": "A bright teenager from a poor home.",
                    "motivations": ["escape poverty"],
                    "arc": "Learns ambition without denial.",
                    "webnovel_role_hook": "underdog scholarship chaser",
                    "progression_function": "turns scarce access into visible academic wins",
                    "speech_style": "Short, guarded, observant.",
                    "education_access": "Public school and borrowed notebooks.",
                    "resources_or_limitations": "No private internet or home TV.",
                    "knowledge_sources": ["school library", "neighbor's phone"],
                }
            ],
            plot_summary="Tare chases a scholarship while protecting his family.",
            chapters=[
                {
                    "chapter_number": 1,
                    "title": "Borrowed Light",
                    "summary": "Tare borrows a book and hides a family debt.",
                    "key_events": ["Tare borrows a textbook"],
                    "characters_involved": ["Tare"],
                    "emotional_arc": "hopeful -> anxious",
                    "opening_hook": "Tare discovers the borrowed book has been marked by a rival.",
                    "progression_reward": "Tare finds a scholarship clue hidden in the book.",
                    "new_question_raised": "Who else knows about the scholarship?",
                    "cliffhanger": "The book's owner arrives at Tare's door.",
                    "reader_emotion_target": "curiosity",
                    "tags_served": ["WEAKTOSTRONG"],
                }
            ],
            serial_arcs=[
                {
                    "arc_number": 1,
                    "title": "The Borrowed Book",
                    "chapter_range": "1-10",
                    "external_goal": "Win the scholarship shortlist.",
                    "central_conflict": "Tare lacks access and time.",
                }
            ],
            release_plan=["Launch with three chapters."],
            retention_strategy=["End each chapter with a new pressure point."],
            long_form_roadmap="Expand through exams, city contests, and family debt.",
            opening_strategy_notes=["Do not start every chapter with a warning."],
            background_constraints=["Tare cannot watch documentaries at home."],
            initial_summary="Tare is resourceful but materially constrained.",
        )

        self.assertEqual(plan.characters[0].speech_style, "Short, guarded, observant.")
        self.assertEqual(plan.background_constraints[0], "Tare cannot watch documentaries at home.")
        self.assertEqual(plan.webnovel_strategy.primary_tags[0], "WEAKTOSTRONG")
        self.assertIn("Tare", plan.launch_chapter_plan.conversion_goal)
        self.assertEqual(plan.chapters[0].cliffhanger, "The book's owner arrives at Tare's door.")
        self.assertEqual(plan.serial_arcs[0].title, "The Borrowed Book")

    def test_launch_chapter_plan_accepts_conversion_fields(self):
        plan = LaunchChapterPlan(
            conversion_goal="Turn browsers into library adds.",
            first_line_strategy="Open on public humiliation.",
            first_200_words_hook="A deadline appears inside the borrowed book.",
            first_scene_pressure="Tare may lose school access if the book is taken.",
            protagonist_snapshot="Brilliant but materially trapped.",
            reader_sympathy_trigger="He hides hunger to protect his sister.",
            special_edge_tease="He remembers every overheard lesson.",
            stakes_lock="The scholarship deadline is tonight.",
            inciting_turn="The rival's note reveals a second exam route.",
            chapter_one_progression_reward="Tare gains the hidden application clue.",
            chapter_one_cliffhanger="The rival knocks on his door.",
            first_five_chapter_promises=["scholarship chase", "rival pressure"],
            tag_delivery_moments=["WEAKTOSTRONG: Tare wins with scarce access"],
            comment_magnet_question="Would you return the book?",
            early_dropoff_risks=["slow setup"],
            revision_checklist=["hook in first 200 words"],
        )

        self.assertEqual(plan.first_five_chapter_promises[0], "scholarship chase")
        self.assertIn("hook", plan.revision_checklist[0])

    def test_character_state_tracks_access_constraints(self):
        state = CharacterState(
            name="Tare",
            knowledge_sources=["school library"],
            resource_constraints=["no home television"],
        )

        self.assertIn("school library", state.knowledge_sources)
        self.assertIn("no home television", state.resource_constraints)


class StoryQualityPromptTests(SimpleTestCase):
    def test_storyteller_prompt_demands_distinct_voice_and_access_planning(self):
        prompt = storyteller_prompt(
            book_title="Dust and Signal",
            book_description="A poor teenager studies by candlelight.",
            num_chapters=3,
            author_profile="plainspoken",
            emotional_guidelines="restrained",
            expert_styles="scene-driven",
        )

        self.assertIn("background_constraints", prompt)
        self.assertIn("speech_style", prompt)
        self.assertIn("opening_strategy_notes", prompt)
        self.assertIn("webnovel_strategy", prompt)
        self.assertIn("launch_chapter_plan", prompt)
        self.assertIn("protagonist_cheat", prompt)
        self.assertIn("serial_arcs", prompt)
        self.assertIn("progression_reward", prompt)
        self.assertIn("TV viewing center", prompt)

    def test_launch_chapter_planner_prompt_builds_conversion_brief(self):
        prompt = launch_chapter_planner_prompt(
            book_title="Dust and Signal",
            book_description="A poor teenager studies by candlelight.",
            story_plan={"title": "Dust and Signal", "chapters": []},
            webnovel_preferences={"tags": ["WEAKTOSTRONG"]},
        )

        self.assertIn("LaunchChapterPlanner", prompt)
        self.assertIn("first_200_words_hook", prompt)
        self.assertIn("comment_magnet_question", prompt)
        self.assertIn("early_dropoff_risks", prompt)

    def test_writer_prompt_includes_repetition_dialogue_and_background_guards(self):
        prompt = writer_prompt(
            chapter_plan={
                "chapter_number": 2,
                "title": "The Borrowed Screen",
                "summary": "Tare hears about a documentary from a neighbor.",
                "key_events": ["Tare visits a neighbor"],
                "characters_involved": ["Tare", "Aunty Bisi"],
                "emotional_arc": "curious -> tense",
                "opening_hook": "Aunty Bisi's TV goes silent during the only useful scene.",
                "progression_reward": "Tare learns one fact that changes his application.",
                "cliffhanger": "The neighbor asks why he is really studying so hard.",
                "tags_served": ["WEAKTOSTRONG"],
                "background_constraints": ["No home TV."],
            },
            author_profile="plainspoken",
            emotional_guidelines="restrained",
            expert_styles="scene-driven",
            running_summary="Chapter 1 ended with Tare losing his textbook.",
            previous_chapter_ending="He shut the door before anyone saw the tears.",
            recent_chapter_summaries=[
                {
                    "chapter_number": 1,
                    "title": "Borrowed Light",
                    "summary": "Tare borrows a book.",
                    "opening_excerpt": "The candle died before Tare finished the first page.",
                }
            ],
            continuity_ledger={
                "characters": [
                    {
                        "name": "Tare",
                        "knowledge_sources": ["school library"],
                        "resource_constraints": ["no home television"],
                    }
                ]
            },
        )

        self.assertIn("RECENT CHAPTER OPENINGS", prompt)
        self.assertIn("copy-paste", prompt)
        self.assertIn("socioeconomic", prompt)
        self.assertIn("knowledge_sources", prompt)
        self.assertIn("documentaries", prompt)
        self.assertIn("progression_reward", prompt)
        self.assertIn("cliffhanger", prompt)
        self.assertIn("Avoid filler", prompt)

    def test_writer_prompt_applies_launch_chapter_plan(self):
        prompt = writer_prompt(
            chapter_plan={"chapter_number": 1, "title": "Borrowed Light"},
            author_profile="plainspoken",
            emotional_guidelines="restrained",
            expert_styles="scene-driven",
            running_summary="",
            previous_chapter_ending="",
            launch_chapter_plan={
                "first_200_words_hook": "Tare finds a deadline in the book.",
                "chapter_one_progression_reward": "Tare gains a scholarship clue.",
                "chapter_one_cliffhanger": "The owner arrives.",
            },
        )

        self.assertIn("LAUNCH CHAPTER PLAN", prompt)
        self.assertIn("first_200_words_hook", prompt)
        self.assertIn("chapter_one_cliffhanger", prompt)

    def test_reviewer_prompt_checks_openings_dialogue_and_background(self):
        prompt = reviewer_prompt(
            {"chapter_number": 2, "title": "The Borrowed Screen"},
            recent_chapter_metadata=[
                {
                    "chapter_number": 1,
                    "title": "Borrowed Light",
                    "opening_excerpt": "The candle died before Tare finished the first page.",
                }
            ],
        )

        self.assertIn("RECENT CHAPTER OPENINGS", prompt)
        self.assertIn("status='revise'", prompt)
        self.assertIn("copy-pasted", prompt)
        self.assertIn("background constraints", prompt)
        self.assertIn("first 200 words", prompt)
        self.assertIn("progression reward", prompt)
        self.assertIn("cliffhanger", prompt)

    def test_reviewer_prompt_adds_first_chapter_review(self):
        prompt = reviewer_prompt(
            {"chapter_number": 1, "title": "Borrowed Light"},
            launch_chapter_plan={
                "first_200_words_hook": "Open with the deadline.",
                "early_dropoff_risks": ["slow setup"],
            },
        )

        self.assertIn("FirstChapterReviewer", prompt)
        self.assertIn("launch_chapter_plan", prompt)
        self.assertIn("early dropoff", prompt)

    def test_perfectionist_prompt_rewrites_specific_quality_failures(self):
        prompt = perfectionist_prompt(
            current_draft="Tare watched the documentary at home.",
            review_feedback="Background plausibility failed.",
            chapter_plan={"chapter_number": 2, "title": "The Borrowed Screen"},
        )

        self.assertIn("If the opening was repetitive", prompt)
        self.assertIn("If dialogue sounded copy-pasted", prompt)
        self.assertIn("background plausibility failed", prompt)
        self.assertIn("Webnovel hook", prompt)
        self.assertIn("progression reward", prompt)

    def test_perfectionist_prompt_repairs_launch_failures(self):
        prompt = perfectionist_prompt(
            current_draft="Tare woke up and thought about school history.",
            review_feedback="Chapter 1 starts too slowly.",
            chapter_plan={"chapter_number": 1, "title": "Borrowed Light"},
            launch_chapter_plan={
                "first_line_strategy": "Open on humiliation.",
                "first_200_words_hook": "The borrowed book has a deadline.",
                "chapter_one_cliffhanger": "The rival arrives.",
            },
        )

        self.assertIn("failed launch-conversion", prompt)
        self.assertIn("first_200_words_hook", prompt)

    def test_continuity_prompt_tracks_access_and_constraints(self):
        prompt = continuity_extractor_prompt(
            chapter_number=2,
            existing_ledger={},
            chapter_plan={"chapter_number": 2, "title": "The Borrowed Screen"},
        )

        self.assertIn("knowledge_sources", prompt)
        self.assertIn("resource_constraints", prompt)
        self.assertIn("documentaries", prompt)
        self.assertIn("reader_open_loops", prompt)
        self.assertIn("progression_milestones", prompt)
