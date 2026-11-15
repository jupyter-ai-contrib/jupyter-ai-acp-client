"""
Pydantic models for legacy/pre-v1 ACP ``models`` payloads.

Some ACP agents (e.g. Kiro, Copilot) advertise or accept models outside the
ACP v1 ``configOptions`` mechanism using the legacy ``models`` payload and
``session/set_model`` JSON-RPC requests.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

log = logging.getLogger(__name__)


def _str_or_none(value: Any) -> Any:
    """Keep a value only if it's a string; otherwise drop it to ``None`` so a
    mistyped field never raises during validation."""
    return value if isinstance(value, str) else None


_TModel = TypeVar("_TModel", bound=BaseModel)


def _safe_parse(model: type[_TModel], data: Any) -> _TModel:
    """
    Validate ``data`` into ``model``, never raising. Each field is already
    tolerant of missing / added / wrong-typed values, so this only backstops a
    wholesale surprise. On any validation error it logs and returns an empty model.
    """
    try:
        return model.model_validate(data)
    except ValidationError:
        log.warning(
            "Ignoring unexpected %s payload shape: %r", model.__name__, data
        )
        return model()


class AcpModelOption(BaseModel):
    """One selectable model in a legacy ``models`` payload."""

    model_config = ConfigDict(
        populate_by_name=True, extra="ignore", protected_namespaces=()
    )

    model_id: Optional[str] = Field(default=None, alias="modelId")
    name: Optional[str] = None
    description: Optional[str] = None

    _coerce_strings = field_validator(
        "model_id", "name", "description", mode="before"
    )(_str_or_none)


class AcpModels(BaseModel):
    """
    The ``models`` block attached to a raw session response: the current
    model plus the list of available models.
    """

    model_config = ConfigDict(
        populate_by_name=True, extra="ignore", protected_namespaces=()
    )

    current_model_id: Optional[str] = Field(default=None, alias="currentModelId")
    available_models: Optional[list[AcpModelOption]] = Field(
        default=None, alias="availableModels"
    )

    _coerce_current = field_validator("current_model_id", mode="before")(_str_or_none)

    @field_validator("available_models", mode="before")
    @classmethod
    def _keep_dict_entries(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [e for e in value if isinstance(e, (dict, AcpModelOption))]
        return None

