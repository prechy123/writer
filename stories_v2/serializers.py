"""DRF serializers for the v2 API.

We use DRF for HTTP plumbing (auth-bypass, content negotiation, error
shapes) but defer the actual schema validation to Pydantic. This avoids
maintaining two parallel field lists.

Each serializer.is_valid() will:
  1. Run the (very loose) DRF parse step.
  2. Hand the dict to its companion Pydantic schema.
  3. Attach either ``self.pydantic`` (the validated model) or DRF errors.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from pydantic import BaseModel, ValidationError
from rest_framework import serializers

from .schemas_v2 import (
    DeepSurvey,
    PastedNotes,
    ProfileV2Input,
    QuickSurvey,
)


class _PydanticPassthroughSerializer(serializers.Serializer):
    """Minimal serializer that delegates all validation to a Pydantic model."""

    pydantic_model: Type[BaseModel] = BaseModel

    def to_internal_value(self, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise serializers.ValidationError({"detail": "Body must be a JSON object."})
        try:
            self.pydantic = self.pydantic_model.model_validate(data)
        except ValidationError as exc:
            errors: Dict[str, Any] = {}
            for err in exc.errors():
                loc = ".".join(str(p) for p in err.get("loc", []))
                errors.setdefault(loc or "_root", []).append(err.get("msg", "invalid"))
            raise serializers.ValidationError(errors)
        return self.pydantic.model_dump()


class ProfileGenerateSerializer(_PydanticPassthroughSerializer):
    pydantic_model = ProfileV2Input


class QuickSurveySerializer(_PydanticPassthroughSerializer):
    pydantic_model = QuickSurvey


class DeepSurveySerializer(_PydanticPassthroughSerializer):
    pydantic_model = DeepSurvey


class PastedNotesSerializer(_PydanticPassthroughSerializer):
    pydantic_model = PastedNotes


class CharacterBiblePatchSerializer(serializers.Serializer):
    """Patch endpoint takes an arbitrary subset of CharacterBibleV2 fields.

    Validation happens by re-validating the merged document on the
    server side after the patch is applied (in views.py).
    """

    def to_internal_value(self, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise serializers.ValidationError({"detail": "Body must be a JSON object."})
        # Bare passthrough; immutable fields are stripped in the view.
        return dict(data)


class WorldBiblePatchSerializer(CharacterBiblePatchSerializer):
    """Same passthrough semantics — patching arbitrary subset of WorldBibleV2."""
    pass
