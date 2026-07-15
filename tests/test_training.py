from __future__ import annotations

from pathlib import Path

import pytest

from gatekeeper.training.cpu_runner import CpuTrainingConfig, build_yolo26_command


def _release_gate_main():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_release_gate", Path(__file__).parents[1] / "scripts" / "check_release_gate.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_release_gate_rejects_candidate_manifest(tmp_path, monkeypatch) -> None:
    main = _release_gate_main()
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"status":"candidate"}', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["check_release_gate", "--manifest", str(manifest)])
    with pytest.raises(SystemExit, match="approved"):
        main()


def test_release_gate_accepts_approved_metrics(tmp_path, monkeypatch) -> None:
    main = _release_gate_main()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"status":"approved","validation":{"exact_code_accuracy":1.0,'
        '"problem_recall":1.0,"problem_false_normal":0.0}}',
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["check_release_gate", "--manifest", str(manifest)])
    assert main() == 0


def test_yolo26_training_command_is_cpu_only() -> None:
    command = build_yolo26_command(
        CpuTrainingConfig(data_yaml=Path("data.yaml"), pretrained=Path("best.pt"))
    )
    assert "device=cpu" in command
    assert "workers=0" in command
    assert "amp=False" in command
    assert "model=best.pt" in command
