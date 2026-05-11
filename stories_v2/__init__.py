"""stories_v2 — humanised web-novel generator.

Lives alongside the v1 ``stories`` app. v1 stays frozen; v2 owns its own
Mongo collections, URL namespace (``/api/v2/``), and execution engine.

The architectural plan is documented at:
    ~/.claude/plans/you-are-a-twenty-wiggly-flame.md
"""

default_app_config = "stories_v2.apps.StoriesV2Config"
