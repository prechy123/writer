from rest_framework import serializers


# ---------------------------------------------------------------------------
# Story serializers
# ---------------------------------------------------------------------------

class StoryCreateSerializer(serializers.Serializer):
    """Validates the POST body for creating a new story."""

    book_title = serializers.CharField(max_length=300)
    book_description = serializers.CharField(max_length=5000)
    num_chapters = serializers.IntegerField(min_value=1, max_value=30)
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


class StoryListItemSerializer(serializers.Serializer):
    """Lightweight representation for the list endpoint."""

    story_id = serializers.CharField(source="_id")
    title = serializers.CharField()
    status = serializers.CharField()
    num_chapters = serializers.IntegerField()
    created_at = serializers.DateTimeField()
    updated_at = serializers.DateTimeField()


class StoryDetailSerializer(serializers.Serializer):
    """Full representation including the manuscript."""

    story_id = serializers.CharField(source="_id")
    title = serializers.CharField()
    description = serializers.CharField()
    status = serializers.CharField()
    num_chapters = serializers.IntegerField()
    created_at = serializers.DateTimeField()
    updated_at = serializers.DateTimeField()
    chapters_completed = serializers.SerializerMethodField()
    final_manuscript = serializers.DictField(allow_null=True)

    def get_chapters_completed(self, obj):
        state = obj.get("state") or {}
        return len(state.get("final_chapters", []))


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
