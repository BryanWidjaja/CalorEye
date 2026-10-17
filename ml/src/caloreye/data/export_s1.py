import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from pycocotools import mask as mask_utils
from ultralytics.data.converter import merge_multi_segment

from caloreye.data.manifest import ACQUISITION_DIR_NAME, WORK_DIR_NAME
from caloreye.data.sam3_label import INSTANCES_NAME, STORE_DIR_NAME
from caloreye.paths import get_data_root_dir
from caloreye.runs import start_run

DATASETS_DIR_NAME = "datasets"
STAGE_ONE_DIR_NAME = "stage1"
LABELS_DIR_NAME = "labels"
IMAGES_PART = "/images/"
LABELS_PART = "/labels/"
MIN_PART_AREA_FRACTION = 0.001
MIN_POLYGON_POINTS = 3
COORDINATE_FORMAT = "{:.6f}"
CLASS_INDEX = 0
CLASS_NAME = "food"
SPLIT_NAMES = ("train", "val")
YAML_NAME = "data.yaml"
SPLITS_GLOB = "splits-*.parquet"

def find_latest_splits(work_dir: Path) -> Path:
    candidates = sorted(work_dir.glob(SPLITS_GLOB))

    if not candidates:
        raise FileNotFoundError(f"expected a splits file in {work_dir}, found none")

    return candidates[-1]

def load_split_lookup(data_root: Path) -> dict[str, str]:
    frame = pl.read_parquet(find_latest_splits(data_root / WORK_DIR_NAME))
    lookup = {}

    image_ids = frame["image_id"].to_list()
    split_names = frame["split"].to_list()

    for image_id, split_name in zip(image_ids, split_names, strict=True):
        lookup[image_id] = split_name

    return lookup

def load_instances(data_root: Path) -> dict:
    instances_path = data_root / STORE_DIR_NAME / INSTANCES_NAME

    return json.loads(instances_path.read_text(encoding="utf-8"))

def decode_annotation_mask(annotation: dict) -> np.ndarray:
    segmentation = annotation["segmentation"]
    encoded = {"size": segmentation["size"], "counts": segmentation["counts"].encode("ascii")}

    return mask_utils.decode(encoded).astype(bool)

def extract_parts(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    parts = []

    for contour in contours:
        area_fraction = cv2.contourArea(contour) / (height * width)

        if area_fraction < MIN_PART_AREA_FRACTION or len(contour) < MIN_POLYGON_POINTS:
            continue

        parts.append(contour.reshape(-1, 2).astype(np.float64))

    return parts

def mask_to_label_line(mask: np.ndarray) -> str | None:
    parts = extract_parts(mask)

    if not parts:
        return None

    height, width = mask.shape

    if len(parts) == 1:
        points = parts[0]
    else:
        points = np.concatenate(merge_multi_segment(parts), axis=0)

    normalized = points / np.array([width, height])
    values = [COORDINATE_FORMAT.format(value) for value in normalized.reshape(-1)]

    return f"{CLASS_INDEX} " + " ".join(values)

def label_path_for(image_path: str) -> str:
    rooted = f"/{image_path}"
    position = rooted.rfind(IMAGES_PART)
    replaced = rooted[:position] + LABELS_PART + rooted[position + len(IMAGES_PART):]

    return Path(replaced).with_suffix(".txt").as_posix().lstrip("/")

def write_label_file(label_path: Path, lines: list[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def group_annotations(payload: dict) -> dict[str, list[dict]]:
    grouped = {}

    for annotation in payload["annotations"]:
        grouped.setdefault(annotation["image_id"], []).append(annotation)

    return grouped

def export_stage_one(data_root: Path) -> list[str]:
    payload = load_instances(data_root)
    grouped = group_annotations(payload)
    splits = load_split_lookup(data_root)
    acquisition_dir = data_root / ACQUISITION_DIR_NAME
    dataset_dir = data_root / DATASETS_DIR_NAME / STAGE_ONE_DIR_NAME
    dataset_dir.mkdir(parents=True, exist_ok=True)
    image_lists = {name: [] for name in SPLIT_NAMES}
    counts = Counter()

    for record in payload["images"]:
        split_name = splits.get(record["image_id"])

        if record["no_food_found"] or split_name is None:
            counts["skipped_images"] += 1
            continue

        lines = []

        for annotation in grouped.get(record["image_id"], []):
            line = mask_to_label_line(decode_annotation_mask(annotation))

            if line is None:
                counts["dropped_instances"] += 1
                continue

            lines.append(line)

        if not lines:
            counts["skipped_images"] += 1
            continue

        write_label_file(acquisition_dir / label_path_for(record["path"]), lines)
        image_lists[split_name].append((acquisition_dir / record["path"]).as_posix())
        counts[f"{split_name}_images"] += 1
        counts[f"{split_name}_instances"] += len(lines)

    return write_dataset_files(dataset_dir, image_lists, counts)

def write_dataset_files(dataset_dir: Path, image_lists: dict, counts: Counter) -> list[str]:
    for split_name, image_paths in image_lists.items():
        list_path = dataset_dir / f"{split_name}.txt"
        list_path.write_text("\n".join(sorted(image_paths)) + "\n", encoding="utf-8")

    yaml_text = "\n".join(
        [
            f"path: {dataset_dir.as_posix()}",
            "train: train.txt",
            "val: val.txt",
            "nc: 1",
            "names:",
            f"  {CLASS_INDEX}: {CLASS_NAME}"
        ]
    )
    yaml_path = dataset_dir / YAML_NAME
    yaml_path.write_text(yaml_text + "\n", encoding="utf-8")

    lines = []

    for key, value in sorted(counts.items()):
        lines.append(f"{key} {value}")

    return lines + [f"wrote {yaml_path}"]

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye data export-s1")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    arguments = parser.parse_args(argv)
    start_run("export-s1", {"data_root": str(arguments.data_root)})

    return export_stage_one(arguments.data_root)
