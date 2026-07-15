from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def copy_required(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"release input missing: {source}")
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_executable(root: Path, stage: Path) -> None:
    separator = ";" if sys.platform == "win32" else ":"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        "gateKeeper",
        "--paths",
        str(root / "src"),
        "--distpath",
        str(stage.parent / "pyinstaller-dist"),
        "--workpath",
        str(stage.parent / "pyinstaller-work"),
        "--specpath",
        str(stage.parent / "pyinstaller-spec"),
        "--add-data",
        f"{root / 'config'}{separator}config",
        "--add-data",
        f"{root / 'docs'}{separator}docs",
        "--add-data",
        f"{root / 'models'}{separator}models",
        "--add-data",
        f"{root / 'plugins'}{separator}plugins",
        "--collect-all",
        "paddleocr",
        "--collect-all",
        "paddlex",
        str(root / "src" / "gatekeeper" / "__main__.py"),
    ]
    subprocess.run(command, check=True, cwd=root)
    built = stage.parent / "pyinstaller-dist" / "gateKeeper"
    copy_required(built, stage)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a self-contained gateKeeper release bundle")
    parser.add_argument("--output", type=Path, default=Path("dist/gateKeeper-bundle"))
    parser.add_argument("--skip-exe", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    stage = output / "gateKeeper"
    if output.exists():
        shutil.rmtree(output)
    stage.mkdir(parents=True)

    detector = root / "models" / "detector.onnx"
    checkpoint = root / "models" / "yolo26s-pcb-pretrained.pt"
    ocr = root / "models" / "ocr"
    manifest = root / "models" / "manifest.json"
    if not detector.is_file():
        raise FileNotFoundError("models/detector.onnx is required in every release")
    if not checkpoint.is_file():
        raise FileNotFoundError("models/yolo26s-pcb-pretrained.pt is required in every release")
    if not ocr.is_dir() or not any(ocr.iterdir()):
        raise FileNotFoundError("models/ocr must contain the bundled OCR pretrained model")
    if not manifest.is_file():
        raise FileNotFoundError("models/manifest.json is required in every release")

    if not args.skip_exe:
        build_executable(root, stage)
    copy_required(root / "config", stage / "config")
    copy_required(root / "docs", stage / "docs")
    copy_required(root / "models", stage / "models")
    copy_required(root / "plugins", stage / "plugins")
    (stage / "watch").mkdir()
    (stage / "archive").mkdir()
    (stage / "logs").mkdir()
    (stage / "overlays").mkdir()

    files: list[dict[str, str | int]] = []
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(stage).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    (stage / "BUILD_MANIFEST.json").write_text(
        json.dumps(
            {
                "product": "gateKeeper",
                "created_at": datetime.now(UTC).isoformat(),
                "self_contained": True,
                "relative_path_root": ".",
                "files": files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    archive = shutil.make_archive(str(output / "gateKeeper"), "zip", root_dir=stage)
    print(f"created {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
