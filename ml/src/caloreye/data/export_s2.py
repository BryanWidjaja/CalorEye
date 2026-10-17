import argparse
import json
import random
from pathlib import Path

import numpy as np
import polars as pl

from caloreye.data.export_s1 import (
    DATASETS_DIR_NAME,
    decode_annotation_mask,
    group_annotations,
    load_instances,
    load_split_lookup,
)
from caloreye.data.sam3_label import NEGATIVES_NAME, STORE_DIR_NAME
from caloreye.paths import get_data_root_dir
from caloreye.runs import start_run

STAGE_TWO_DIR_NAME = "stage2"
CROPS_NAME = "crops.parquet"
CLASS_SOURCE_IDS = ("6", "5", "39", "B1", "11", "1")
NOT_FOOD_INDEX = 6
ORIGIN_WHOLE = "whole"
ORIGIN_GT_BOX = "gt_box"
ORIGIN_NOT_FOOD = "bukan_makanan"
MAX_NEGATIVE_FOOD_OVERLAP = 0.2
PATCH_MIN_FRACTION = 0.25
PATCH_MAX_FRACTION = 0.5
PATCH_ATTEMPTS = 30
PATCH_SEED = 0
FULL_BOX = (0.0, 0.0, 1.0, 1.0)
SCHEMA = {
    "image_id": pl.String,
    "path": pl.String,
    "source_id": pl.String,
    "x0": pl.Float32,
    "y0": pl.Float32,
    "x1": pl.Float32,
    "y1": pl.Float32,
    "dish_idx": pl.Int32,
    "origin": pl.String,
    "split": pl.String
}

def make_row(record: dict, box: tuple, dish_idx: int, origin: str, split_name: str) -> dict:
    return {
        "image_id": record["image_id"],
        "path": record["path"],
        "source_id": record["source_id"],
        "x0": box[0],
        "y0": box[1],
        "x1": box[2],
        "y1": box[3],
        "dish_idx": dish_idx,
        "origin": origin,
        "split": split_name
    }

def normalize_bbox(bbox: list[int], record: dict) -> tuple[float, float, float, float]:
    left, top, box_width, box_height = bbox
    width = record["width"]
    height = record["height"]

    return (left / width, top / height, (left + box_width) / width, (top + box_height) / height)

def build_dish_rows(payload: dict, grouped: dict, splits: dict) -> list[dict]:
    rows = []

    for record in payload["images"]:
        split_name = splits.get(record["image_id"])

        if record["no_food_found"] or split_name is None:
            continue

        dish_idx = CLASS_SOURCE_IDS.index(record["source_id"])
        rows.append(make_row(record, FULL_BOX, dish_idx, ORIGIN_WHOLE, split_name))
        largest = max(grouped[record["image_id"]], key=lambda annotation: annotation["area"])
        box = normalize_bbox(largest["bbox"], record)
        rows.append(make_row(record, box, dish_idx, ORIGIN_GT_BOX, split_name))

    return rows

def build_negative_rows(
        negatives: list[dict],
        records: dict,
        splits: dict,
        limit: int,
        rng: random.Random
    ) -> list[dict]:
    eligible = []

    for annotation in negatives:
        record = records.get(annotation["image_id"])
        is_clear = annotation["food_overlap"] <= MAX_NEGATIVE_FOOD_OVERLAP

        if record is None or not is_clear or record["image_id"] not in splits:
            continue

        eligible.append(annotation)

    rng.shuffle(eligible)
    rows = []

    for annotation in eligible[:limit]:
        record = records[annotation["image_id"]]
        box = normalize_bbox(annotation["bbox"], record)
        split_name = splits[record["image_id"]]
        rows.append(make_row(record, box, NOT_FOOD_INDEX, ORIGIN_NOT_FOOD, split_name))

    return rows

def find_patch_box(
        food_union: np.ndarray,
        rng: random.Random
    ) -> tuple[float, float, float, float] | None:
    height, width = food_union.shape

    attempts = 0

    while attempts < PATCH_ATTEMPTS:
        attempts += 1
        side_fraction = rng.uniform(PATCH_MIN_FRACTION, PATCH_MAX_FRACTION)
        side = int(side_fraction * min(height, width))
        left = rng.randint(0, width - side)
        top = rng.randint(0, height - side)

        if not food_union[top:top + side, left:left + side].any():
            return (left / width, top / height, (left + side) / width, (top + side) / height)

    return None

def build_patch_rows(
        payload: dict,
        grouped: dict,
        splits: dict,
        limit: int,
        rng: random.Random
    ) -> list[dict]:
    candidates = []

    for record in payload["images"]:
        if not record["no_food_found"]:
            candidates.append(record)

    rng.shuffle(candidates)
    rows = []

    for record in candidates:
        if len(rows) >= limit:
            break

        if record["image_id"] not in splits:
            continue

        union = np.zeros((record["height"], record["width"]), dtype=bool)

        for annotation in grouped[record["image_id"]]:
            union |= decode_annotation_mask(annotation)

        box = find_patch_box(union, rng)

        if box is not None:
            split_name = splits[record["image_id"]]
            rows.append(make_row(record, box, NOT_FOOD_INDEX, ORIGIN_NOT_FOOD, split_name))

    return rows

def export_stage_two(data_root: Path) -> list[str]:
    payload = load_instances(data_root)
    grouped = group_annotations(payload)
    splits = load_split_lookup(data_root)
    negatives_path = data_root / STORE_DIR_NAME / NEGATIVES_NAME
    negatives = json.loads(negatives_path.read_text(encoding="utf-8"))["annotations"]
    records = {record["image_id"]: record for record in payload["images"]}
    rng = random.Random(PATCH_SEED)
    dish_rows = build_dish_rows(payload, grouped, splits)
    whole_rows = [row for row in dish_rows if row["origin"] == ORIGIN_WHOLE]
    whole_count = len(whole_rows)
    limit = round(whole_count / len(CLASS_SOURCE_IDS))
    negative_rows = build_negative_rows(negatives, records, splits, limit, rng)
    patch_rows = build_patch_rows(payload, grouped, splits, limit - len(negative_rows), rng)
    frame = pl.DataFrame(dish_rows + negative_rows + patch_rows, schema=SCHEMA)
    output_dir = data_root / DATASETS_DIR_NAME / STAGE_TWO_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_dir / CROPS_NAME)
    counts = frame.group_by(["origin", "split"]).len().sort(["origin", "split"])
    lines = []

    for origin, split_name, count in counts.iter_rows():
        lines.append(f"{origin} {split_name} {count}")

    return lines + [f"negative boxes {len(negative_rows)}, background patches {len(patch_rows)}"]

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye data export-s2")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    arguments = parser.parse_args(argv)
    start_run("export-s2", {"data_root": str(arguments.data_root)})

    return export_stage_two(arguments.data_root)
