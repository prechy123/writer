from django.urls import path

from . import views

urlpatterns = [
    # Stories
    path("stories/", views.list_stories_view, name="story-list"),
    path("stories/create/", views.create_story, name="story-create"),
    path("stories/<str:story_id>/", views.get_story_detail, name="story-detail"),
    path("stories/<str:story_id>/continue/", views.continue_story_view, name="story-continue"),
    path("stories/<str:story_id>/hide/", views.hide_story_view, name="story-hide"),
    path("stories/<str:story_id>/unhide/", views.unhide_story_view, name="story-unhide"),
    # Profiles (generate once, reuse for every story)
    path("profiles/", views.list_profiles_view, name="profile-list"),
    path("profiles/generate/", views.generate_profile_view, name="profile-generate"),
    path("profiles/<str:profile_id>/", views.get_profile_view, name="profile-detail"),
]
