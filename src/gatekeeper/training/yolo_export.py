from __future__ import annotations

import json
import shutil
from pathlib import Path


def export_grouped_yolo_dataset(processed_root: Path) -> Path:
    """Export reviewed COCO rectangles to a deterministic group-separated YOLO dataset."""

    detector = processed_root / "detector"
    annotations_path = detector / "annotations.json"
    groups_path = processed_root / "manifests" / "label-groups.json"
    if not annotations_path.is_file() or not groups_path.is_file():
        raise FileNotFoundError("reviewed annotations and panel/lot/recipe groups are required")
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    group_records = json.loads(groups_path.read_text(encoding="utf-8"))["records"]
    groups = {str(item["image"]): str(item["group"]) for item in group_records}
    images = {int(item["id"]): item for item in payload["images"]}
    categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    if set(categories.values()) != {"fpcb_surface", "code_roi"}:
        raise ValueError("the exported dataset must contain fpcb_surface and code_roi only")
    missing_groups = [
        Path(str(item["file_name"])).name
        for item in images.values()
        if Path(str(item["file_name"])).name not in groups
    ]
    if missing_groups:
        raise ValueError(f"missing group assignment for {len(missing_groups)} image(s)")
    unique_groups = sorted(set(groups.values()))
    if len(unique_groups) < 3:
        raise ValueError("at least three independent panel, lot, or recipe groups are required")
    group_split = _split_groups(unique_groups)
    yolo_root = detector / "yolo"
    if yolo_root.exists():
        shutil.rmtree(yolo_root)
    annotations: dict[int, list[dict[str, object]]] = {image_id: [] for image_id in images}
    for item in payload["annotations"]:
        annotations[int(item["image_id"])].append(item)
    class_ids = {"fpcb_surface": 0, "code_roi": 1}
    split_records: list[dict[str, str]] = []
    for image_id, image in images.items():
        filename = Path(str(image["file_name"])).name
        split = group_split[groups[filename]]
        source = detector / str(image["file_name"])
        target_image = yolo_root / "images" / split / filename
        target_label = yolo_root / "labels" / split / f"{Path(filename).stem}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_image)
        width, height = int(image["width"]), int(image["height"])
        lines: list[str] = []
        present: set[str] = set()
        for item in annotations[image_id]:
            category = categories[int(item["category_id"])]
            x, y, box_width, box_height = (float(value) for value in item["bbox"])
            lines.append(
                f"{class_ids[category]} {(x + box_width / 2) / width:.8f} "
                f"{(y + box_height / 2) / height:.8f} {box_width / width:.8f} "
                f"{box_height / height:.8f}"
            )
            present.add(category)
        if present != set(class_ids):
            raise ValueError(f"image {filename} does not contain both required rectangles")
        target_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        split_records.append({"image": filename, "panel_id": groups[filename], "split": split})
    data_yaml = detector / "data.yaml"
    data_yaml.write_text(
        "path: yolo\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: fpcb_surface\n  1: code_roi\n",
        encoding="utf-8",
    )
    manifest = processed_root / "manifests" / "panel-splits.json"
    manifest.write_text(json.dumps({"images": split_records}, indent=2), encoding="utf-8")
    return data_yaml


def _split_groups(groups: list[str]) -> dict[str, str]:
    if len(groups) < 10:
        names = ("train", "val", "test")
        return {group: names[index % len(names)] for index, group in enumerate(groups)}
    output: dict[str, str] = {}
    for index, group in enumerate(groups):
        remainder = index % 10
        output[group] = "train" if remainder < 7 else "val" if remainder < 9 else "test"
    return output
