"""URL routes for /api/v2/.

Phase 1: health + provider probe.
Phase 2: profile generation, Quick/Deep story creation, paste parser,
         bible CRUD.
Phase 9 will add: /stories/<id>/, /stories/<id>/skeleton/,
                  /stories/<id>/stream/, /continue/, /regenerate-*.
"""

from django.urls import path

from . import views

urlpatterns = [
    # Smoke
    path("health/", views.health, name="v2-health"),
    path("providers/probe/", views.provider_probe, name="v2-providers-probe"),

    # Profiles v2
    path("profiles/", views.list_profiles_view, name="v2-profiles-list"),
    path("profiles/generate/", views.generate_profile_view, name="v2-profile-generate"),
    path("profiles/<str:profile_id>/", views.get_profile_view, name="v2-profile-detail"),
    path("profiles/<str:profile_id>/hide/", views.hide_profile_view, name="v2-profile-hide"),
    path("profiles/<str:profile_id>/unhide/", views.unhide_profile_view, name="v2-profile-unhide"),

    # Stories
    path("stories/", views.list_stories_view, name="v2-stories-list"),
    path("stories/quick/", views.create_story_quick, name="v2-story-quick"),
    path("stories/deep/", views.create_story_deep, name="v2-story-deep"),
    path("stories/parse/", views.parse_pasted_view, name="v2-story-parse"),
    path("stories/<str:story_id>/", views.get_story_reader, name="v2-story-detail"),
    path("stories/<str:story_id>/start/", views.start_story_view, name="v2-story-start"),
    path("stories/<str:story_id>/continue/", views.continue_story_view, name="v2-story-continue"),
    path("stories/<str:story_id>/stream/", views.stream_story_events, name="v2-story-stream"),
    path("stories/<str:story_id>/hide/", views.hide_story_view, name="v2-story-hide"),
    path("stories/<str:story_id>/unhide/", views.unhide_story_view, name="v2-story-unhide"),

    # Edit / regenerate
    path("stories/<str:story_id>/regenerate-scene/", views.regenerate_scene_view, name="v2-story-regen-scene"),
    path("stories/<str:story_id>/regenerate-from/", views.regenerate_from_view, name="v2-story-regen-from"),
    path("stories/<str:story_id>/scenes/<int:chapter_idx>/<int:scene_idx>/", views.patch_scene_view, name="v2-story-patch-scene"),

    # Skeleton tree (engine internals — frontend "Skeleton" button)
    path("stories/<str:story_id>/skeleton/", views.skeleton_view, name="v2-skeleton"),
    path("stories/<str:story_id>/skeleton/arc/", views.skeleton_arc_view, name="v2-skeleton-arc"),
    path("stories/<str:story_id>/skeleton/world/", views.skeleton_world_view, name="v2-skeleton-world"),
    path("stories/<str:story_id>/skeleton/characters/", views.skeleton_characters_view, name="v2-skeleton-characters"),
    path("stories/<str:story_id>/skeleton/chapters/<int:chapter_idx>/beats/", views.skeleton_chapter_beats_view, name="v2-skeleton-chapter-beats"),
    path("stories/<str:story_id>/skeleton/chapters/<int:chapter_idx>/scenes/", views.skeleton_chapter_scenes_view, name="v2-skeleton-chapter-scenes"),
    path("stories/<str:story_id>/skeleton/events/", views.skeleton_events_view, name="v2-skeleton-events"),

    # Bibles
    path(
        "bibles/<str:story_id>/characters/",
        views.list_characters_view,
        name="v2-bible-characters-list",
    ),
    path(
        "bibles/<str:story_id>/character/<str:character_id>/",
        views.get_character_view,
        name="v2-bible-character-get",
    ),
    path(
        "bibles/<str:story_id>/character/<str:character_id>/patch/",
        views.patch_character_view,
        name="v2-bible-character-patch",
    ),
    path(
        "bibles/<str:story_id>/world/",
        views.get_world_view,
        name="v2-bible-world-get",
    ),
    path(
        "bibles/<str:story_id>/world/patch/",
        views.patch_world_view,
        name="v2-bible-world-patch",
    ),
]
