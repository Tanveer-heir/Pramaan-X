"""Allowlisted, investigation-scoped artifact publication."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import mimetypes
from pathlib import Path
import re
from threading import Lock

from .models import ArtifactReference


ARTIFACT_KEY = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class RegisteredArtifact:
    path: Path
    reference: ArtifactReference


class ArtifactRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._entries: dict[tuple[str, str], RegisteredArtifact] = {}
        self._lock = Lock()

    def investigation_dir(self, investigation_id: str) -> Path:
        path = (self.root / investigation_id).resolve()
        if path.parent != self.root:
            raise ValueError("Invalid investigation identifier")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def register(self, investigation_id: str, key: str, path: Path) -> ArtifactReference:
        if not ARTIFACT_KEY.fullmatch(key):
            raise ValueError("Invalid artifact key")
        resolved = path.resolve()
        expected_parent = self.investigation_dir(investigation_id)
        if expected_parent not in resolved.parents or not resolved.is_file():
            raise ValueError("Artifact is outside its investigation directory")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        reference = ArtifactReference(
            url=f"/api/v1/investigations/{investigation_id}/artifacts/{key}",
            media_type=media_type,
            sha256=digest,
        )
        with self._lock:
            self._entries[(investigation_id, key)] = RegisteredArtifact(resolved, reference)
        return reference

    def resolve(self, investigation_id: str, key: str) -> RegisteredArtifact | None:
        if not ARTIFACT_KEY.fullmatch(key):
            return None
        with self._lock:
            entry = self._entries.get((investigation_id, key))
        if entry is None or not entry.path.is_file():
            return None
        expected_parent = (self.root / investigation_id).resolve()
        if expected_parent not in entry.path.resolve().parents:
            return None
        return entry
