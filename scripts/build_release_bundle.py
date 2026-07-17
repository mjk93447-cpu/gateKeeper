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


PRODUCT_NAME = "Manufacturing Junction gateKeeper AI Vision"
PRODUCT_SLUG = "Manufacturing-Junction-gateKeeper-AI-Vision"
TRAINING_RUNNER = "Manufacturing Junction gateKeeper Training"


def run_pyinstaller(
    python_executable: str,
    root: Path,
    stage: Path,
    name: str,
    entrypoint: Path,
    collect_all: list[str],
    *,
    onefile: bool = False,
    exclude: list[str] | None = None,
    collect_submodules: list[str] | None = None,
    hiddenimports: list[str] | None = None,
    metadata: list[str] | None = None,
) -> None:
    command = [
        python_executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile" if onefile else "--onedir",
        "--name",
        name,
        "--paths",
        str(root / "src"),
        "--distpath",
        str(stage.parent / "pyinstaller-dist"),
        "--workpath",
        str(stage.parent / "pyinstaller-work"),
        "--specpath",
        str(stage.parent / "pyinstaller-spec"),
    ]
    for package in collect_all:
        command.extend(["--collect-all", package])
    for package in collect_submodules or []:
        command.extend(["--collect-submodules", package])
    for package in hiddenimports or []:
        command.extend(["--hidden-import", package])
    for distribution in metadata or []:
        command.extend(["--copy-metadata", distribution])
    for package in exclude or []:
        command.extend(["--exclude-module", package])
    # PyInstaller expects its positional entrypoint after all collection
    # options. Keeping it last also makes this reproducible with the manual
    # frozen-runtime verification command.
    command.append(str(entrypoint))
    subprocess.run(command, check=True, cwd=root)
    suffix = ".exe" if sys.platform == "win32" and onefile else ""
    built = stage.parent / "pyinstaller-dist" / f"{name}{suffix}"
    copy_required(built, stage)


def build_executables(root: Path, stage: Path, training_python: str) -> None:
    run_pyinstaller(
        sys.executable,
        root,
        stage,
        PRODUCT_NAME,
        root / "src" / "gatekeeper" / "__main__.py",
        # PaddleOCR delegates prediction construction to PaddleX at runtime.
        # PaddleX loads several backend modules dynamically, so its complete
        # package must be collected for a frozen offline executable.
        [
            "aistudio_sdk",
            "bidi",
            "colorlog",
            "filelock",
            "huggingface_hub",
            "httpx",
            "imagesize",
            "matplotlib",
            "modelscope",
            "paddle",
            "paddleocr",
            "paddlex",
            "pandas",
            "prettytable",
            "pynvml",
            "requests",
            "ruamel",
            "ujson",
        ],
        exclude=["pytest", "tensorflow", "ultralytics"],
        # `idna` imports this CPython extension dynamically through requests.
        # Include it explicitly so offline frozen PaddleX startup is reliable.
        hiddenimports=[
            "torch",
            "torch.multiprocessing",
            "torch.distributed",
            "unicodedata",
        ],
        metadata=[
            "opencv-contrib-python",
            "pypdfium2",
            "python-bidi",
        ],
    )
    run_pyinstaller(
        training_python,
        root,
        stage,
        TRAINING_RUNNER,
        root / "src" / "gatekeeper" / "training" / "embedded_runner.py",
        ["ultralytics"],
        onefile=True,
    )


def copy_release_models(root: Path, stage: Path) -> None:
    source = root / "models"
    destination = stage / "models"
    for filename in ("detector.onnx", "yolo26s-pcb-pretrained.pt", "manifest.json"):
        copy_required(source / filename, destination / filename)
    selected_ocr = source / "ocr" / "official_models" / "en_PP-OCRv4_mobile_rec"
    if not selected_ocr.is_dir():
        matches = sorted((source / "ocr").glob("**/en_PP-OCRv4_mobile_rec/inference.yml"))
        if not matches:
            raise FileNotFoundError("the en_PP-OCRv4_mobile_rec OCR model is required")
        selected_ocr = matches[0].parent
    copy_required(selected_ocr, destination / "ocr" / "en_PP-OCRv4_mobile_rec")
    shutil.rmtree(destination / "ocr" / "en_PP-OCRv4_mobile_rec" / ".cache", ignore_errors=True)


def add_corresponding_source(root: Path, stage: Path, version: str) -> None:
    source = stage / "source"
    source.mkdir(parents=True, exist_ok=True)
    archive = source / f"{PRODUCT_SLUG}-source-v{version}.zip"
    try:
        subprocess.run(
            ["git", "archive", "--format=zip", f"--output={archive}", "HEAD"],
            check=True,
            cwd=root,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("unable to create the required corresponding-source archive") from exc
    (source / "SOURCE_OFFER.txt").write_text(
        "This installer includes the release-specific corresponding source archive.\n"
        "See LICENSE and NOTICE for the AGPL-3.0-or-later terms.\n",
        encoding="utf-8",
    )


def set_bundle_version(stage: Path, version: str) -> None:
    manifest = stage / "models" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_version"] = version
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a self-contained gateKeeper release bundle")
    parser.add_argument("--output", type=Path, default=Path("dist/gateKeeper-bundle"))
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument(
        "--training-python",
        default=sys.executable,
        help="Python interpreter with the offline CPU training dependencies installed",
    )
    parser.add_argument("--skip-exe", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    version = args.version.removeprefix("v")
    stage = output / PRODUCT_SLUG
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
        build_executables(root, stage, args.training_python)
    copy_required(root / "config", stage / "config")
    copy_required(root / "docs", stage / "docs")
    copy_release_models(root, stage)
    set_bundle_version(stage, version)
    copy_required(root / "plugins", stage / "plugins")
    copy_required(root / "LICENSE", stage / "LICENSE")
    copy_required(root / "NOTICE", stage / "NOTICE")
    add_corresponding_source(root, stage, version)
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
                "product": PRODUCT_NAME,
                "version": version,
                "created_at": datetime.now(UTC).isoformat(),
                "self_contained": True,
                "relative_path_root": ".",
                "files": files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    archive = shutil.make_archive(str(output / PRODUCT_SLUG), "zip", root_dir=stage)
    print(f"created {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
