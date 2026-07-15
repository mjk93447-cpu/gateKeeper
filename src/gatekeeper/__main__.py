from __future__ import annotations

import sys


def main() -> int:
    try:
        from gatekeeper.ui.app import run
    except ImportError as exc:
        print(
            "Desktop dependencies are missing. Install with: "
            'uv pip install -e ".[desktop]"',
            file=sys.stderr,
        )
        print(f"detail: {exc}", file=sys.stderr)
        return 2
    return run()


if __name__ == "__main__":
    raise SystemExit(main())

