from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_VALID_CODE = re.compile(r"^[A-Z0-9]{4}$")


def _normalize(value: str) -> str:
    code = re.sub(r"[\s_-]+", "", value.upper())
    if not _VALID_CODE.fullmatch(code):
        raise ValueError("codes must contain exactly four characters from A-Z and 0-9")
    return code


def _o_zero_key(code: str) -> str:
    return code.replace("O", "0")


@dataclass(frozen=True, slots=True)
class CodeRecipe:
    """Persisted normal/problem code registry used by inspection and training checks."""

    normal_codes: tuple[str, ...] = ("HJ04",)
    problem_codes: tuple[str, ...] = ("HJ05",)
    active_normal_code: str = "HJ04"

    def __post_init__(self) -> None:
        normal = tuple(dict.fromkeys(_normalize(code) for code in self.normal_codes))
        problem = tuple(dict.fromkeys(_normalize(code) for code in self.problem_codes))
        active = _normalize(self.active_normal_code)
        if not normal:
            raise ValueError("at least one normal code is required")
        if active not in normal:
            raise ValueError("active normal code must be registered as normal")
        normal_keys = {_o_zero_key(code) for code in normal}
        if normal_keys & {_o_zero_key(code) for code in problem}:
            raise ValueError("normal and problem codes cannot overlap after O/0 correction")
        object.__setattr__(self, "normal_codes", normal)
        object.__setattr__(self, "problem_codes", problem)
        object.__setattr__(self, "active_normal_code", active)

    @property
    def all_codes(self) -> frozenset[str]:
        return frozenset((*self.normal_codes, *self.problem_codes))

    @classmethod
    def load(cls, path: str | Path) -> CodeRecipe:
        recipe_path = Path(path)
        if not recipe_path.is_file():
            return cls()
        payload = json.loads(recipe_path.read_text(encoding="utf-8"))
        return cls(
            normal_codes=tuple(payload.get("normal_codes", ("HJ04",))),
            problem_codes=tuple(payload.get("problem_codes", ("HJ05",))),
            active_normal_code=str(payload.get("active_normal_code", "HJ04")),
        )

    def save(self, path: str | Path) -> None:
        recipe_path = Path(path)
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text(
            json.dumps(
                {
                    "normal_codes": list(self.normal_codes),
                    "problem_codes": list(self.problem_codes),
                    "active_normal_code": self.active_normal_code,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
