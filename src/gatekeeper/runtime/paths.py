from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    """All paths are resolved from the install bundle, never from CWD."""

    root: Path

    @classmethod
    def discover(cls) -> RuntimePaths:
        override = os.environ.get("GATEKEEPER_HOME")
        if override:
            return cls(Path(override).expanduser().resolve())
        if getattr(sys, "frozen", False):
            return cls(Path(sys.executable).resolve().parent)
        return cls(Path(__file__).resolve().parents[3])

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def config(self) -> Path:
        return self.path("config", "default.json")

    @property
    def code_recipe(self) -> Path:
        return self.path("config", "code_recipe.json")

    @property
    def frame_gate_background(self) -> Path:
        return self.path("config", "frame_gate_empty_background.npy")

    @property
    def models(self) -> Path:
        return self.path("models")

    @property
    def watch(self) -> Path:
        return self.path("watch")

    @property
    def archive(self) -> Path:
        return self.path("archive")

    @property
    def logs(self) -> Path:
        return self.path("logs")

    @property
    def overlays(self) -> Path:
        return self.path("overlays")

    @property
    def docs(self) -> Path:
        return self.path("docs")

    @property
    def plugins(self) -> Path:
        return self.path("plugins")

    @property
    def training_runner(self) -> Path:
        return self.path("Manufacturing Junction gateKeeper Training.exe")
