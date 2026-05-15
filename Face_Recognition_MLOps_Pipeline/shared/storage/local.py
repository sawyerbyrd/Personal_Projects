"""Local filesystem storage — the only concrete backend in-repo."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import BinaryIO

from shared.storage.base import StorageBackend
from shared.storage.models import ArtifactRef


class LocalFilesystemStorage(StorageBackend):
    """
    Maps logical keys under ``storage_root``.

    URI format: ``local://<relative_key>`` where *relative_key* uses forward slashes,
    e.g. ``local://preprocessed/7d2c.../manifest.json``.

    TODO(infra): ``MountedFilesystemStorage`` can subclass this class and override only
    ``_physical_path`` if the mount uses the same directory layout as ``storage_root``.

    TODO(infra): For multi-tenant keys, prefix ``relative_key`` with a tenant id here.
    """

    scheme = "local"

    def __init__(self, storage_root: Path) -> None:
        self.storage_root = storage_root.resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def _physical_path(self, relative_key: str) -> Path:
        rel = relative_key.replace("\\", "/").lstrip("/")
        path = (self.storage_root / rel).resolve()
        try:
            path.relative_to(self.storage_root)
        except ValueError as exc:
            raise ValueError(f"Refusing to escape storage_root: {relative_key}") from exc
        return path

    def _to_uri(self, relative_key: str) -> str:
        key = relative_key.replace("\\", "/").lstrip("/")
        return f"{self.scheme}://{key}"

    def save_bytes(
        self,
        relative_key: str,
        data: bytes,
        *,
        content_type: str | None = None,
    ) -> ArtifactRef:
        path = self._physical_path(relative_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return ArtifactRef(
            uri=self._to_uri(relative_key),
            logical_path=relative_key,
            content_type=content_type,
            byte_size=len(data),
        )

    def save_file(self, relative_key: str, source_path: Path) -> ArtifactRef:
        dest = self._physical_path(relative_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)
        size = dest.stat().st_size
        return ArtifactRef(
            uri=self._to_uri(relative_key),
            logical_path=relative_key,
            byte_size=size,
        )

    def open_read_stream(self, ref: ArtifactRef) -> BinaryIO:
        path = self._physical_path(self._uri_to_key(ref.uri))
        return path.open("rb")

    def load_file(self, ref: ArtifactRef) -> bytes:
        path = self._physical_path(self._uri_to_key(ref.uri))
        return path.read_bytes()

    def exists(self, ref: ArtifactRef) -> bool:
        return self._physical_path(self._uri_to_key(ref.uri)).exists()

    def list(self, relative_prefix: str) -> list[ArtifactRef]:
        base = self._physical_path(relative_prefix)
        if not base.exists():
            return []
        out: list[ArtifactRef] = []
        for p in base.rglob("*"):
            if p.is_file():
                rel = p.relative_to(self.storage_root).as_posix()
                out.append(
                    ArtifactRef(
                        uri=self._to_uri(rel),
                        logical_path=rel,
                        byte_size=p.stat().st_size,
                    )
                )
        return sorted(out, key=lambda r: r.uri)

    def resolve_uri(self, uri: str) -> ArtifactRef:
        key = self._uri_to_key(uri)
        p = self._physical_path(key)
        size = p.stat().st_size if p.exists() else None
        return ArtifactRef(uri=uri, logical_path=key, byte_size=size)

    def ref_from_key(self, relative_key: str) -> ArtifactRef:
        key = relative_key.replace("\\", "/").lstrip("/")
        return ArtifactRef(uri=self._to_uri(key), logical_path=key)

    def get_local_path_if_available(self, ref: ArtifactRef) -> Path | None:
        return self._physical_path(self._uri_to_key(ref.uri))

    @staticmethod
    def _uri_to_key(uri: str) -> str:
        if "://" not in uri:
            raise ValueError(f"Invalid artifact URI: {uri}")
        scheme, rest = uri.split("://", 1)
        if scheme != LocalFilesystemStorage.scheme:
            raise ValueError(f"URI scheme {scheme!r} is not supported by LocalFilesystemStorage")
        return rest.lstrip("/")
