from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CpuTrainingConfig:
    data_yaml: Path
    pretrained: Path
    output_dir: Path = Path("runs/gatekeeper")
    image_size: int = 640
    epochs: int = 100
    batch: int = 2
    patience: int = 20
    workers: int = 0
    seed: int = 42


def build_yolo26_command(config: CpuTrainingConfig) -> list[str]:
    """Build a reproducible CPU-only Ultralytics training command."""

    return [
        "yolo",
        "detect",
        "train",
        f"model={config.pretrained}",
        f"data={config.data_yaml}",
        f"imgsz={config.image_size}",
        f"epochs={config.epochs}",
        f"batch={config.batch}",
        "device=cpu",
        f"workers={config.workers}",
        f"patience={config.patience}",
        "amp=False",
        "cache=disk",
        f"seed={config.seed}",
        f"project={config.output_dir}",
        "name=yolo26s_fpcb_cpu",
    ]
