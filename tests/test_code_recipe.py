from __future__ import annotations

import pytest

from gatekeeper.domain import CodeRecipe, DecisionEngine, InspectionInput, InspectionState
from gatekeeper.training.code_labels import validate_ocr_labels


def test_default_recipe_keeps_hj04_normal_and_hj05_problem(tmp_path) -> None:
    recipe = CodeRecipe.load(tmp_path / "missing.json")
    assert recipe.normal_codes == ("HJ04",)
    assert recipe.problem_codes == ("HJ05",)
    assert recipe.active_normal_code == "HJ04"


def test_recipe_persists_added_codes_and_drives_decision(tmp_path) -> None:
    path = tmp_path / "code_recipe.json"
    CodeRecipe(("HJ04", "AB12"), ("HJ05", "CD34"), "AB12").save(path)
    recipe = CodeRecipe.load(path)
    result = DecisionEngine().decide(
        InspectionInput(
            expected_code=recipe.active_normal_code,
            problem_codes=frozenset(recipe.problem_codes),
            detected=True,
            detector_confidence=0.99,
            ocr_text="CD34",
            ocr_confidence=0.99,
        )
    )
    assert recipe.normal_codes == ("HJ04", "AB12")
    assert result.state is InspectionState.PROBLEM


def test_recipe_rejects_o_zero_collision() -> None:
    with pytest.raises(ValueError, match="overlap"):
        CodeRecipe(("HO04",), ("H004",), "HO04")


def test_ocr_labels_must_match_recipe(tmp_path) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"image")
    labels = tmp_path / "labels.txt"
    labels.write_text("sample.png\tHJ04\nsample.png\tZZ99\n", encoding="utf-8")
    errors, counts = validate_ocr_labels(labels, CodeRecipe().all_codes)
    assert counts["HJ04"] == 1
    assert any("ZZ99" in error for error in errors)
