"""Per-use-case data-protection policy, enforced by the core for every request.

The rules come from the manifest, so a use case with health data (sickness
absence) gets strict checks while a greenhouse use case runs with a minimal
configuration. These are engineering measures for data minimisation (GDPR
art. 25); whether a client may process the data at all is the client's call.
"""

from __future__ import annotations

from typing import Any

from predictor.core.manifest import ManifestError, Policy
from predictor.core.validation import FieldError


def check_record(policy: Policy, features: Any) -> list[FieldError]:
    if not isinstance(features, dict):
        return []
    errors: list[FieldError] = []
    forbidden = {name.lower() for name in policy.forbidden_fields}
    for name in features:
        if name.lower() in forbidden:
            errors.append(FieldError(name, "this field is forbidden by the data-protection policy of this use case"))

    rule = policy.min_group_size
    if rule is not None:
        size = features.get(rule.field)
        if isinstance(size, (int, float)) and not isinstance(size, bool) and size < rule.min:
            errors.append(
                FieldError(
                    rule.field,
                    f"groups smaller than {rule.min} may not be processed (re-identification risk)",
                )
            )
    return errors


def check_schema(use_case_id: str, policy: Policy, feature_schema: dict[str, Any]) -> None:
    """A model may never have been trained on a forbidden field."""
    forbidden = {name.lower() for name in policy.forbidden_fields}
    clash = sorted(name for name in feature_schema["properties"] if name.lower() in forbidden)
    if clash:
        raise ManifestError(f"{use_case_id}: the feature schema contains forbidden fields {clash}")
    if policy.min_group_size and policy.min_group_size.field not in feature_schema["properties"]:
        raise ManifestError(
            f"{use_case_id}: min_group_size.field {policy.min_group_size.field!r} is not a model feature"
        )
