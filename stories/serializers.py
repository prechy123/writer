from rest_framework import serializers


# ---------------------------------------------------------------------------
# Story serializers
# ---------------------------------------------------------------------------

class StoryCreateSerializer(serializers.Serializer):
    """Validates the POST body for creating a new story."""

    book_title = serializers.CharField(max_length=300)
    book_description = serializers.CharField(max_length=5000)
    num_chapters = serializers.IntegerField(min_value=1, max_value=30)
    initial_chapters = serializers.IntegerField(
        required=False,
        default=None,
        allow_null=True,
        min_value=1,
        max_value=30,
        help_text=(
            "How many chapters to draft in this initial run. The Storyteller "
            "always plans the full book upfront; only the Writer loop is "
            "bounded. Defaults to settings.DEFAULT_INITIAL_CHAPTERS (3), "
            "clamped to [1, num_chapters]. If initial_chapters >= num_chapters "
            "the book is written in one run and the Publisher runs immediately."
        ),
    )
    profile_id = serializers.CharField(
        required=False,
        default=None,
        allow_null=True,
        help_text=(
            "ID of a stored profile. If provided, Phase 1 agents are skipped "
            "and the stored author_profile / emotional_guidelines / expert_styles "
            "are injected directly. If provided but not found → 400 error. "
            "If omitted → generic Phase 1 agents run."
        ),
    )


class StoryContinueSerializer(serializers.Serializer):
    """Validates the POST body for /api/stories/<id>/continue/."""

    additional_chapters = serializers.IntegerField(
        min_value=1,
        max_value=30,
        help_text=(
            "How many more chapters to draft in this batch. Must not exceed "
            "the number of chapters still unwritten (num_chapters - "
            "current_chapter_index); otherwise the request returns 400."
        ),
    )


class StoryListItemSerializer(serializers.Serializer):
    """Lightweight representation for the list endpoint."""

    story_id = serializers.CharField(source="_id")
    title = serializers.CharField()
    status = serializers.CharField()
    num_chapters = serializers.IntegerField()
    created_at = serializers.DateTimeField()
    updated_at = serializers.DateTimeField()


class StoryDetailSerializer(serializers.Serializer):
    """Full representation including the manuscript and continuity context."""

    story_id = serializers.CharField(source="_id")
    title = serializers.CharField()
    description = serializers.CharField()
    status = serializers.CharField()
    num_chapters = serializers.IntegerField()
    created_at = serializers.DateTimeField()
    updated_at = serializers.DateTimeField()
    chapters_completed = serializers.SerializerMethodField()
    current_chapter_index = serializers.SerializerMethodField()
    target_chapter_index = serializers.SerializerMethodField()
    chapters_remaining = serializers.SerializerMethodField()
    chapter_metadata = serializers.SerializerMethodField()
    continuity_ledger = serializers.SerializerMethodField()
    batch_log = serializers.ListField(required=False, default=list)
    final_manuscript = serializers.DictField(allow_null=True)

    def _state(self, obj):
        return obj.get("state") or {}

    def get_chapters_completed(self, obj):
        return len(self._state(obj).get("final_chapters", []))

    def get_current_chapter_index(self, obj):
        return int(self._state(obj).get("current_chapter_index", 0) or 0)

    def get_target_chapter_index(self, obj):
        state = self._state(obj)
        return int(state.get("target_chapter_index") or obj.get("num_chapters", 0))

    def get_chapters_remaining(self, obj):
        state = self._state(obj)
        total = int(obj.get("num_chapters", 0) or 0)
        current = int(state.get("current_chapter_index", 0) or 0)
        return max(0, total - current)

    def get_chapter_metadata(self, obj):
        """Return per-chapter metadata without the full chapter text."""
        return self._state(obj).get("chapter_metadata", []) or []

    def get_continuity_ledger(self, obj):
        return self._state(obj).get("continuity_ledger", {}) or {}


# ---------------------------------------------------------------------------
# Profile serializers
# ---------------------------------------------------------------------------

class ProfileGenerateSerializer(serializers.Serializer):
    """Validates the POST body for generating a friend profile.

    Accepts rich biographical context so the Profiler / Empath / Masterclass
    agents can build an accurate voice model in a single pass.
    """

    name = serializers.CharField(max_length=200)

    # Rich biographical fields (all optional — provide as many as possible)
    background = serializers.CharField(
        required=False, default="", allow_blank=True,
        help_text="Who the person is, where they grew up, formative experiences.",
    )
    personality = serializers.CharField(
        required=False, default="", allow_blank=True,
        help_text="Temperament, what pisses them off, sense of humour, fears.",
    )
    communication_style = serializers.CharField(
        required=False, default="", allow_blank=True,
        help_text="How they talk, text, write — short/long sentences, slang, formality.",
    )
    interests_and_values = serializers.CharField(
        required=False, default="", allow_blank=True,
        help_text="What they care about deeply, hobbies, passions.",
    )
    quirks = serializers.CharField(
        required=False, default="", allow_blank=True,
        help_text="Habits, catchphrases, pet peeves, recurring mannerisms.",
    )
    additional_context = serializers.CharField(
        required=False, default="", allow_blank=True,
        help_text="Anything else that helps capture this person's essence.",
    )

    # Expert style (optional — if provided, skips the Masterclass agent)
    expert_style = serializers.CharField(
        required=False, default="", allow_blank=True,
        help_text="Style notes from a professional writer/expert. If provided, used as the basis for expert_styles instead of generating via the Masterclass agent.",
    )
    expert_writing_sample = serializers.CharField(
        required=False, default="", allow_blank=True,
        help_text="A writing sample from the expert/professional. Used alongside expert_style to extract additional stylistic cues.",
    )

    # Writing samples (optional but highly recommended)
    writing_samples = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        default=list,
        help_text="Actual text written by this person — messages, posts, stories, etc.",
    )
