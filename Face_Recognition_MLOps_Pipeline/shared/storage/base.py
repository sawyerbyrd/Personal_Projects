"""Abstract storage backend — data plane: paths/URIs only."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

from shared.storage.models import ArtifactRef


class StorageBackend(ABC):
    """
    Storage abstraction. Business logic must depend on this type, not on ``Path`` directly.

    TODO(infra): Implement ``MountedFilesystemStorage`` delegating to the same URI layout
    as ``LocalFilesystemStorage`` but with a configurable POSIX root (NFS / SMB mount).

    TODO(infra): Implement ``S3Storage`` using boto3 or aioboto3; map ``ArtifactRef.uri``
    to ``s3://`` keys while keeping method signatures identical.
    """

    scheme: str

    @abstractmethod
    def save_bytes(self, relative_key: str, data: bytes, *, content_type: str | None = None) -> ArtifactRef:
        """Persist raw bytes at a logical key under the configured artifact layout."""

    @abstractmethod
    def save_file(self, relative_key: str, source_path: Path) -> ArtifactRef:
        """Copy an existing file from ``source_path`` into storage at ``relative_key``."""

    @abstractmethod
    def open_read_stream(self, ref: ArtifactRef) -> BinaryIO:
        """Return a binary stream for reading the artifact."""

    @abstractmethod
    def load_file(self, ref: ArtifactRef) -> bytes:
        """Load entire artifact into memory (use only for small objects)."""

    @abstractmethod
    def exists(self, ref: ArtifactRef) -> bool:
        """Return True if the artifact is available."""

    @abstractmethod
    def list(self, relative_prefix: str) -> list[ArtifactRef]:
        """List artifacts under a logical prefix (e.g. ``preprocessed/job_id/``)."""

    @abstractmethod
    def resolve_uri(self, uri: str) -> ArtifactRef:
        """Parse a URI string into an ``ArtifactRef`` for this backend."""

    @abstractmethod
    def ref_from_key(self, relative_key: str) -> ArtifactRef:
        """Return a reference for a logical key (file may not exist yet)."""

    @abstractmethod
    def get_local_path_if_available(self, ref: ArtifactRef) -> Path | None:
        """
        If the artifact is directly readable as a local path, return it.

        Remote backends should return None (or a temp path after download — not implemented).
        """
