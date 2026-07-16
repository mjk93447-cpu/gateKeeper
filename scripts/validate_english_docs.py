from __future__ import annotations

import re
from pathlib import Path

NON_ENGLISH = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af\u4e00-\u9fff]")
ROOTS = (Path("README.md"), Path("docs"), Path("data"), Path("training"), Path("models"))


def files_to_check(root: Path) -> list[Path]:
    files: list[Path] = []
    for item in ROOTS:
        path = root / item
        if path.is_file() and path.suffix.lower() in {".md", ".json", ".yml", ".yaml"}:
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in {".md", ".json", ".yml", ".yaml"}
            )
    return sorted(set(files))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors: list[str] = []
    for path in files_to_check(root):
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            if NON_ENGLISH.search(line):
                errors.append(
                    f"{path.relative_to(root)}:{line_number}: non-English script detected"
                )
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
