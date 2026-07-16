from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    name: str
    path: Path
    sha256: str
    architecture: str


class ModelRegistry:
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.payload: dict[str, Any] = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )

    def artifact(self, name: str) -> ModelArtifact:
        record = self.payload[name]
        path = self.manifest_path.parent / record["path"]
        return ModelArtifact(
            name=name,
            path=path,
            sha256=str(record["sha256"]).lower(),
            architecture=str(record["architecture"]),
        )

    def verify(self, name: str) -> ModelArtifact:
        artifact = self.artifact(name)
        if not artifact.path.is_file():
            raise FileNotFoundError(f"{name} artifact not found: {artifact.path}")
        digest = hashlib.sha256(artifact.path.read_bytes()).hexdigest()
        if digest != artifact.sha256:
            raise ValueError(
                f"{name} SHA-256 mismatch: expected {artifact.sha256}, received {digest}"
            )
        return artifact
