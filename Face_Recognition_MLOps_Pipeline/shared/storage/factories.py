"""Storage backend construction from settings."""

from __future__ import annotations

from pathlib import Path

from shared.config.settings import StorageSettings
from shared.storage.base import StorageBackend
from shared.storage.local import LocalFilesystemStorage


def get_storage(settings: StorageSettings) -> StorageBackend:
    """Return the configured storage backend.

    Switch backends by setting STORAGE_BACKEND in the environment:
      - ``local``  — local filesystem (default, used for dev/demo)
      - ``s3``     — AWS S3; requires S3_BUCKET + standard AWS credentials
    """
    backend = settings.storage_backend
    if backend == "local":
        return LocalFilesystemStorage(Path(settings.storage_root))
    if backend == "s3":
        from shared.storage.s3 import S3Storage

        if not settings.s3_bucket:
            raise ValueError("S3_BUCKET must be set when STORAGE_BACKEND=s3")
        return S3Storage(
            bucket=settings.s3_bucket,
            prefix=settings.s3_prefix,
            region=settings.s3_region or None,
            endpoint_url=settings.s3_endpoint_url or None,
        )
    raise NotImplementedError(f"STORAGE_BACKEND={backend!r} is not supported.")
