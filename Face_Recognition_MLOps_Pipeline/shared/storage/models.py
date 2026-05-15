"""Artifact reference — shared between storage backends and control-plane DTOs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ArtifactRef(BaseModel):
    """
    Opaque artifact handle exchanged between services.

    Local backend URIs use ``local://`` (see ``shared/storage/local.py``).

    TODO(infra): S3 backend may use ``s3://bucket/key`` while preserving this model.
    """

    uri: str = Field(..., description="Storage-specific URI for the artifact.")
    logical_path: str | None = Field(
        default=None,
        description="Relative path under storage root for debugging and mounted FS parity.",
    )
    content_type: str | None = Field(default=None)
    byte_size: int | None = Field(default=None)
