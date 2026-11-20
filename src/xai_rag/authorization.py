"""Trusted, static authorization policy for isolated tenant indexes."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True, slots=True)
class AccessScope:
    tenant: str
    sources: tuple[str, ...]
    can_ingest: bool


def granted_ingestion_paths(root: Path, source: Path, scope: AccessScope) -> list[Path]:
    """Resolve a request lexically against grants, without probing the filesystem."""
    import os

    path = Path(os.path.abspath(source))
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise PermissionError("ingestion is not authorized") from exc
    selected = [
        item
        for item in scope.sources
        if relative == "." or item == relative or item.startswith(relative + "/")
    ]
    if not scope.can_ingest or not selected:
        raise PermissionError("ingestion is not authorized")
    return [root / item for item in selected]


@dataclass(frozen=True, slots=True)
class Policy:
    entries: tuple[tuple[str, AccessScope], ...]

    @property
    def tenants(self) -> frozenset[str]:
        return frozenset(scope.tenant for _, scope in self.entries)

    def authenticate(self, key: str) -> AccessScope | None:
        if len(key.encode("utf-8")) < 32:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        result = None
        for expected, scope in self.entries:
            if secrets.compare_digest(digest, expected):
                result = scope
        return result


def load_policy(path: str) -> Policy:
    """Load a server-controlled policy; no malformed or ambiguous grants accepted."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = data["principals"]
        if not isinstance(entries, list) or not 1 <= len(entries) <= 128:
            raise ValueError("invalid principal count")
        result = []
        seen: set[str] = set()
        for entry in entries:
            digest, tenant = entry["key_sha256"], entry["tenant"]
            sources, role = entry["sources"], entry["role"]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("invalid key digest")
            if (
                digest in seen
                or not isinstance(tenant, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tenant)
            ):
                raise ValueError("duplicate credential or invalid tenant")
            if (
                role not in {"reader", "ingester"}
                or not isinstance(sources, list)
                or len(sources) > 1000
            ):
                raise ValueError("invalid role or sources")
            for source in sources:
                if (
                    not isinstance(source, str)
                    or not source
                    or "\\" in source
                    or PurePosixPath(source).is_absolute()
                    or ".." in PurePosixPath(source).parts
                    or str(PurePosixPath(source)) != source
                ):
                    raise ValueError("invalid source identity")
            if len(sources) != len(set(sources)):
                raise ValueError("duplicate source identity")
            seen.add(digest)
            result.append((digest, AccessScope(tenant, tuple(sources), role == "ingester")))
        return Policy(tuple(result))
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid authorization policy") from exc


def tenant_store_name(base: str, tenant: str) -> str:
    """Allocate new store names that cannot expose legacy unscoped data."""
    digest = hashlib.sha256(f"{base}\0{tenant}".encode()).hexdigest()[:32]
    return f"xai-scoped-{digest}"
