from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RectangleAnnotation:
    category: str
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.category not in {"fpcb_surface", "code_roi"}:
            raise ValueError("category must be fpcb_surface or code_roi")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("rectangle width and height must be positive")


class AnnotationStore:
    """Persist reviewed rectangular ROI masks as COCO and OCR training labels."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.detector_root = root / "detector"
        self.ocr_root = root / "ocr"
        self.annotations_path = self.detector_root / "annotations.json"

    def save(
        self,
        image_path: Path,
        image_size: tuple[int, int],
        code: str,
        annotations: tuple[RectangleAnnotation, ...],
        group: str,
    ) -> Path:
        if {item.category for item in annotations} != {"fpcb_surface", "code_roi"}:
            raise ValueError("one fpcb_surface and one code_roi rectangle are required")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        if not group.strip():
            raise ValueError("panel, lot, or recipe group is required")

        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()[:12]
        suffix = image_path.suffix.lower() or ".png"
        name = f"{image_path.stem}_{digest}{suffix}"
        detector_image = self.detector_root / "images" / name
        detector_image.parent.mkdir(parents=True, exist_ok=True)
        if not detector_image.exists():
            shutil.copy2(image_path, detector_image)

        payload = self._read_payload()
        existing = next(
            (item for item in payload["images"] if item["file_name"] == f"images/{name}"), None
        )
        image_id = int(existing["id"]) if existing else self._next_id(payload["images"])
        payload["images"] = [item for item in payload["images"] if item["id"] != image_id]
        payload["annotations"] = [
            item for item in payload["annotations"] if item["image_id"] != image_id
        ]
        payload["images"].append(
            {"id": image_id, "file_name": f"images/{name}", "width": width, "height": height}
        )
        categories = {item["name"]: int(item["id"]) for item in payload["categories"]}
        annotation_id = self._next_id(payload["annotations"])
        for annotation in annotations:
            if annotation.x < 0 or annotation.y < 0:
                raise ValueError("rectangle coordinates cannot be negative")
            if annotation.x + annotation.width > width or annotation.y + annotation.height > height:
                raise ValueError("rectangle must stay inside the image")
            x, y, w, h = annotation.x, annotation.y, annotation.width, annotation.height
            payload["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": categories[annotation.category],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
                }
            )
            annotation_id += 1
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        self.annotations_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        code_roi = next(item for item in annotations if item.category == "code_roi")
        self._write_ocr_crop(detector_image, name, code, code_roi)
        self._write_group(name, group)
        return self.annotations_path

    def _read_payload(self) -> dict[str, list[dict[str, object]]]:
        if self.annotations_path.is_file():
            return json.loads(self.annotations_path.read_text(encoding="utf-8"))
        return {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "fpcb_surface"},
                {"id": 1, "name": "code_roi"},
            ],
        }

    @staticmethod
    def _next_id(records: list[dict[str, object]]) -> int:
        return max((int(item["id"]) for item in records), default=0) + 1

    def _write_ocr_crop(
        self, image: Path, name: str, code: str, annotation: RectangleAnnotation
    ) -> None:
        import cv2

        source = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f"unable to read image for OCR crop: {image}")
        crop = source[
            annotation.y : annotation.y + annotation.height,
            annotation.x : annotation.x + annotation.width,
        ]
        ocr_name = f"{Path(name).stem}_code.png"
        destination = self.ocr_root / "images" / ocr_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), crop):
            raise OSError(f"unable to write OCR crop: {destination}")
        labels = self.ocr_root / "train.txt"
        entries: dict[str, str] = {}
        if labels.is_file():
            for line in labels.read_text(encoding="utf-8").splitlines():
                relative, value = line.rsplit("\t", 1)
                entries[relative] = value
        entries[f"images/{ocr_name}"] = code
        labels.write_text(
            "\n".join(f"{path}\t{value}" for path, value in sorted(entries.items())) + "\n",
            encoding="utf-8",
        )

    def _write_group(self, name: str, group: str) -> None:
        path = self.root / "manifests" / "label-groups.json"
        records = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"records": []}
        )
        records["records"] = [item for item in records["records"] if item["image"] != name]
        records["records"].append({"image": name, "group": group.strip()})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
