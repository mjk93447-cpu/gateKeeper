from __future__ import annotations

from pathlib import Path

from gatekeeper.training.cpu_runner import CpuTrainingConfig, build_yolo26_command


def test_yolo26_training_command_is_cpu_only() -> None:
    command = build_yolo26_command(
        CpuTrainingConfig(data_yaml=Path("data.yaml"), pretrained=Path("best.pt"))
    )
    assert "device=cpu" in command
    assert "workers=0" in command
    assert "amp=False" in command
    assert "model=best.pt" in command
