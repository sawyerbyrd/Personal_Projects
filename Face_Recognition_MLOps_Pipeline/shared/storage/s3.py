"""S3 storage backend — swap in by setting STORAGE_BACKEND=s3."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from shared.storage.base import StorageBackend
from shared.storage.models import ArtifactRef

if TYPE_CHECKING:
    import boto3 as boto3_type


class S3Storage(StorageBackend):
    """
    Maps logical keys to S3 objects under ``bucket/prefix/``.

    URI format: ``s3://bucket-name/prefix/relative/key``

    Switch from local to S3 by setting:
        STORAGE_BACKEND=s3
        S3_BUCKET=my-bucket
        S3_PREFIX=my-prefix          # optional
        S3_REGION=us-east-1          # optional
        S3_ENDPOINT_URL=...          # optional, for LocalStack / MinIO
    AWS credentials are resolved via the standard boto3 chain
    (env vars AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, ~/.aws/credentials, IAM role).
    """

    scheme = "s3"

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        import boto3

        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._s3 = boto3.client(
            "s3",
            region_name=region or None,
            endpoint_url=endpoint_url or None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _full_key(self, relative_key: str) -> str:
        key = relative_key.replace("\\", "/").lstrip("/")
        return f"{self._prefix}/{key}" if self._prefix else key

    def _to_uri(self, relative_key: str) -> str:
        return f"s3://{self._bucket}/{self._full_key(relative_key)}"

    def _uri_to_relative_key(self, uri: str) -> str:
        if not uri.startswith("s3://"):
            raise ValueError(f"Not an S3 URI: {uri!r}")
        rest = uri[len("s3://"):]
        _bucket, _, full_key = rest.partition("/")
        if self._prefix and full_key.startswith(self._prefix + "/"):
            return full_key[len(self._prefix) + 1:]
        return full_key

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def save_bytes(
        self,
        relative_key: str,
        data: bytes,
        *,
        content_type: str | None = None,
    ) -> ArtifactRef:
        extra: dict = {}
        if content_type:
            extra["ContentType"] = content_type
        self._s3.put_object(Bucket=self._bucket, Key=self._full_key(relative_key), Body=data, **extra)
        return ArtifactRef(
            uri=self._to_uri(relative_key),
            logical_path=relative_key,
            content_type=content_type,
            byte_size=len(data),
        )

    def save_file(self, relative_key: str, source_path: Path) -> ArtifactRef:
        data = source_path.read_bytes()
        return self.save_bytes(relative_key, data)

    def open_read_stream(self, ref: ArtifactRef) -> BinaryIO:
        key = self._full_key(self._uri_to_relative_key(ref.uri))
        response = self._s3.get_object(Bucket=self._bucket, Key=key)
        return response["Body"]

    def load_file(self, ref: ArtifactRef) -> bytes:
        key = self._full_key(self._uri_to_relative_key(ref.uri))
        response = self._s3.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    def exists(self, ref: ArtifactRef) -> bool:
        from botocore.exceptions import ClientError

        key = self._full_key(self._uri_to_relative_key(ref.uri))
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    def list(self, relative_prefix: str) -> list[ArtifactRef]:
        prefix = self._full_key(relative_prefix)
        if not prefix.endswith("/"):
            prefix += "/"
        paginator = self._s3.get_paginator("list_objects_v2")
        refs: list[ArtifactRef] = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                full_key = obj["Key"]
                rel = full_key[len(self._prefix) + 1:] if self._prefix else full_key
                refs.append(
                    ArtifactRef(
                        uri=f"s3://{self._bucket}/{full_key}",
                        logical_path=rel,
                        byte_size=obj.get("Size"),
                    )
                )
        return sorted(refs, key=lambda r: r.uri)

    def resolve_uri(self, uri: str) -> ArtifactRef:
        rel = self._uri_to_relative_key(uri)
        try:
            head = self._s3.head_object(Bucket=self._bucket, Key=self._full_key(rel))
            size = head.get("ContentLength")
        except Exception:
            size = None
        return ArtifactRef(uri=uri, logical_path=rel, byte_size=size)

    def ref_from_key(self, relative_key: str) -> ArtifactRef:
        return ArtifactRef(uri=self._to_uri(relative_key), logical_path=relative_key)

    def get_local_path_if_available(self, ref: ArtifactRef) -> Path | None:
        return None
