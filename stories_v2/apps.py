from django.apps import AppConfig


class StoriesV2Config(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "stories_v2"
    verbose_name = "Stories v2 (humanised engine)"

    def ready(self) -> None:
        from . import mongo

        try:
            mongo.ensure_indexes()
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "stories_v2: ensure_indexes() failed at startup — "
                "indexes will be created lazily on first use.",
                exc_info=True,
            )
