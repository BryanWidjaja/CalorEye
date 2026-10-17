import argparse
import json
import logging
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils

from caloreye.data.clean import find_latest_flags
from caloreye.data.manifest import (
    ACQUISITION_DIR_NAME,
    DATE_FORMAT,
    WORK_DIR_NAME,
    convert_to_rgb,
)
from caloreye.paths import get_data_root_dir
from caloreye.runs import start_run
from caloreye.taxonomy import PILOT_SOURCE_IDS

PROMPT_FOOD = "food"
PROMPT_SAUCE = "dipping sauce"
PROMPT_PLATE = "plate"
PROMPT_BOWL = "bowl"
PROMPTS = [PROMPT_FOOD, PROMPT_SAUCE, PROMPT_PLATE, PROMPT_BOWL]
ROLE_FOOD = "food"
ROLE_NEGATIVE = "negative"
ROLE_BY_PROMPT = {
    PROMPT_FOOD: ROLE_FOOD,
    PROMPT_SAUCE: ROLE_FOOD,
    PROMPT_PLATE: ROLE_NEGATIVE,
    PROMPT_BOWL: ROLE_NEGATIVE
}
MIN_CONF_BY_PROMPT = {
    PROMPT_FOOD: 0.5,
    PROMPT_SAUCE: 0.4,
    PROMPT_PLATE: 0.5,
    PROMPT_BOWL: 0.5
}
PREDICTOR_CONF = 0.3
MIN_AREA_FRACTION = 0.01
MAX_AREA_FRACTION = 0.95
IOU_DUPLICATE_LIMIT = 0.6
CONTAINMENT_LIMIT = 0.9
MAX_FOOD_MASKS = 8
MAX_NEGATIVE_MASKS = 4
MAX_LONG_SIDE = 1280
POLYGON_EPSILON_FRACTION = 0.002
MIN_POLYGON_POINTS = 3
MIN_PART_AREA_FRACTION = 0.0005
SHARD_SIZE = 100
SAM3_CACHE_RELATIVE_PATH = Path(".cache/huggingface/hub/models--facebook--sam3/snapshots")
SAM3_WEIGHTS_PATTERN = "*/sam3.pt"
STORE_DIR_NAME = "store"
SHARD_DIR_NAME = "sam3_shards"
INSTANCES_NAME = "instances.json"
NEGATIVES_NAME = "negatives.json"
IMAGES_NAME = "images.json"
CATEGORY_FOOD = 1
SOURCE_NAME = "sam3"
LABEL_SCOPE = "image"
SAMPLE_COUNT = 50
SHEET_COLUMNS = 10
TILE_SIZE = 160
SHEET_SEED = 0
OUTLINE_THICKNESS = 2
FOOD_COLOR = (0, 255, 0)
SAUCE_COLOR = (255, 0, 255)
SHEET_QUALITY = 85
MIN_FREE_BYTES = 5 * 1024 ** 3
JSON_INDENT = 2
SECONDS_PER_HOUR = 3600
BYTE_MAX = 255

@dataclass
class KeptMask:
    index: int
    prompt: str
    conf: float
    role: str
    area_fraction: float
    food_overlap: float = 0.0

def mask_overlap_counts(first: np.ndarray, second: np.ndarray) -> tuple[int, int, int]:
    intersection = int(np.logical_and(first, second).sum())
    first_area = int(first.sum())
    second_area = int(second.sum())

    return intersection, first_area, second_area

def is_redundant(candidate: np.ndarray, kept_masks: list[np.ndarray], prompt: str) -> str | None:
    for kept in kept_masks:
        intersection, candidate_area, kept_area = mask_overlap_counts(candidate, kept)
        union = candidate_area + kept_area - intersection

        if union > 0 and intersection / union > IOU_DUPLICATE_LIMIT:
            return "duplicate"

        contained = candidate_area > 0 and intersection / candidate_area >= CONTAINMENT_LIMIT

        if contained and prompt != PROMPT_SAUCE:
            return "contained"

    return None

def filter_masks(
        masks: np.ndarray,
        confs: np.ndarray,
        class_ids: np.ndarray
    ) -> tuple[list[KeptMask], Counter]:
    stats = Counter()
    kept = []
    kept_by_role = {ROLE_FOOD: [], ROLE_NEGATIVE: []}
    caps = {ROLE_FOOD: MAX_FOOD_MASKS, ROLE_NEGATIVE: MAX_NEGATIVE_MASKS}

    for index in np.argsort(-confs):
        prompt = PROMPTS[int(class_ids[index])]
        role = ROLE_BY_PROMPT[prompt]
        mask = masks[index]
        area_fraction = float(mask.sum()) / mask.size

        if confs[index] < MIN_CONF_BY_PROMPT[prompt]:
            stats["low_conf"] += 1
            continue

        if area_fraction < MIN_AREA_FRACTION or area_fraction > MAX_AREA_FRACTION:
            stats["area"] += 1
            continue

        reason = is_redundant(mask, kept_by_role[role], prompt)

        if reason is None and len(kept_by_role[role]) >= caps[role]:
            reason = "cap"

        if reason is not None:
            stats[reason] += 1
            continue

        kept_by_role[role].append(mask)
        kept.append(KeptMask(int(index), prompt, float(confs[index]), role, area_fraction))

    attach_food_overlap(kept, masks, kept_by_role[ROLE_FOOD])

    return kept, stats

def attach_food_overlap(
        kept: list[KeptMask],
        masks: np.ndarray,
        food_masks: list[np.ndarray]
    ) -> None:
    if not food_masks:
        return

    food_union = np.logical_or.reduce(food_masks)

    for item in kept:
        if item.role != ROLE_NEGATIVE:
            continue

        negative = masks[item.index]
        overlap = np.logical_and(negative, food_union).sum()
        item.food_overlap = float(overlap) / max(int(negative.sum()), 1)

def mask_to_rle(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))

    size = [int(value) for value in encoded["size"]]

    return {"size": size, "counts": encoded["counts"].decode("ascii")}

def mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    min_area = MIN_PART_AREA_FRACTION * mask.size

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        epsilon = POLYGON_EPSILON_FRACTION * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)

        if len(simplified) < MIN_POLYGON_POINTS:
            continue

        polygons.append([float(value) for value in simplified.reshape(-1)])

    return polygons

def rasterize_polygons(polygons: list[list[float]], height: int, width: int) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)

    for polygon in polygons:
        points = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [points], 1)

    return canvas.astype(bool)

def build_annotation(item: KeptMask, mask: np.ndarray, image_id: str, annotation_id: str) -> dict:
    rows, columns = np.where(mask)
    left, top = int(columns.min()), int(rows.min())
    box_width = int(columns.max()) - left + 1
    box_height = int(rows.max()) - top + 1

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": CATEGORY_FOOD,
        "segmentation": mask_to_rle(mask),
        "polygons": mask_to_polygons(mask),
        "bbox": [left, top, box_width, box_height],
        "area": int(mask.sum()),
        "source": SOURCE_NAME,
        "reviewed": False,
        "prompt": item.prompt,
        "conf": round(item.conf, 4),
        "label_scope": LABEL_SCOPE,
        "area_fraction": round(item.area_fraction, 4),
        "food_overlap": round(item.food_overlap, 4)
    }

def load_model_pixels(image_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(image_path) as image:
        upright = ImageOps.exif_transpose(image)
        rgb_image = convert_to_rgb(upright)
        original_size = rgb_image.size
        scale = MAX_LONG_SIDE / max(original_size)

        if scale < 1.0:
            new_size = (round(original_size[0] * scale), round(original_size[1] * scale))
            rgb_image = rgb_image.resize(new_size, Image.LANCZOS)

        rgb_array = np.asarray(rgb_image)

    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR), original_size

def find_sam3_weights() -> Path:
    cache_dir = Path.home() / SAM3_CACHE_RELATIVE_PATH
    matches = sorted(cache_dir.glob(SAM3_WEIGHTS_PATTERN))

    if not matches:
        raise FileNotFoundError(f"expected sam3.pt under {cache_dir}, found none")

    return matches[0]

def build_predictor():
    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = {
        "conf": PREDICTOR_CONF,
        "task": "segment",
        "mode": "predict",
        "model": str(find_sam3_weights()),
        "quantize": 16,
        "save": False,
        "verbose": False
    }

    return SAM3SemanticPredictor(overrides=overrides)

def predict_masks(predictor, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictor.set_image(pixels)
    result = predictor(text=PROMPTS)[0]

    if result.masks is None:
        empty = np.zeros((0, *pixels.shape[:2]), dtype=bool)

        return empty, np.zeros(0), np.zeros(0, dtype=np.int64)

    masks = result.masks.data.cpu().numpy().astype(bool)
    confs = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(np.int64)

    return masks, confs, class_ids

def label_one_image(
        predictor,
        image_path: Path,
        record: dict
    ) -> tuple[dict, list[dict], list[dict], Counter]:
    pixels, original_size = load_model_pixels(image_path)
    masks, confs, class_ids = predict_masks(predictor, pixels)
    kept, stats = filter_masks(masks, confs, class_ids)
    instances = []
    negatives = []

    for position, item in enumerate(kept):
        annotation_id = f"{record['image_id']}-{position}"
        annotation = build_annotation(item, masks[item.index], record["image_id"], annotation_id)

        if item.role == ROLE_FOOD:
            instances.append(annotation)
        else:
            negatives.append(annotation)

    image_record = {
        "image_id": record["image_id"],
        "path": record["path"],
        "source_id": record["source_id"],
        "original_width": original_size[0],
        "original_height": original_size[1],
        "width": int(pixels.shape[1]),
        "height": int(pixels.shape[0]),
        "n_food_masks": len(instances),
        "n_negatives": len(negatives),
        "no_food_found": len(instances) == 0
    }

    return image_record, instances, negatives, stats

def read_done_ids(shard_dir: Path) -> set[str]:
    done = set()

    for shard_path in shard_dir.glob("shard-*.json"):
        shard = json.loads(shard_path.read_text(encoding="utf-8"))

        for image_record in shard["images"]:
            done.add(image_record["image_id"])

    return done

def write_shard(shard_dir: Path, shard: dict) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    index = len(list(shard_dir.glob("shard-*.json")))
    shard_path = shard_dir / f"shard-{index:05d}.json"
    shard_path.write_text(json.dumps(shard), encoding="utf-8")

def select_targets(data_root: Path, source_ids: list[str]) -> list[dict]:
    work_dir = data_root / WORK_DIR_NAME
    kept = pl.read_parquet(find_latest_flags(work_dir)).filter(pl.col("keep"))
    targets = kept.filter(pl.col("source_id").is_in(source_ids))
    targets = targets.unique(subset=["image_id"], keep="first").sort("path")

    return targets.select(["image_id", "path", "source_id"]).to_dicts()

def merge_shards(shard_dir: Path, store_dir: Path) -> dict[str, int]:
    images = []
    instances = []
    negatives = []

    for shard_path in sorted(shard_dir.glob("shard-*.json")):
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        images.extend(shard["images"])
        instances.extend(shard["instances"])
        negatives.extend(shard["negatives"])

    categories = [{"id": CATEGORY_FOOD, "name": "food"}]
    store_dir.mkdir(parents=True, exist_ok=True)
    instances_payload = {"images": images, "annotations": instances, "categories": categories}
    (store_dir / INSTANCES_NAME).write_text(json.dumps(instances_payload), encoding="utf-8")
    negatives_payload = {"annotations": negatives, "categories": categories}
    (store_dir / NEGATIVES_NAME).write_text(json.dumps(negatives_payload), encoding="utf-8")
    (store_dir / IMAGES_NAME).write_text(json.dumps(images), encoding="utf-8")

    return {"images": len(images), "instances": len(instances), "negatives": len(negatives)}

def run_labelling(
        data_root: Path,
        source_ids: list[str],
        limit: int | None
    ) -> tuple[Counter, int, float]:
    free_bytes = shutil.disk_usage(data_root).free

    if free_bytes < MIN_FREE_BYTES:
        raise OSError(f"expected at least {MIN_FREE_BYTES} free bytes, found {free_bytes}")

    store_dir = data_root / STORE_DIR_NAME
    shard_dir = store_dir / SHARD_DIR_NAME
    done_ids = read_done_ids(shard_dir)
    targets = []

    for target in select_targets(data_root, source_ids):
        if target["image_id"] not in done_ids:
            targets.append(target)

    if limit is not None:
        targets = targets[:limit]

    predictor = build_predictor()
    stats = Counter()
    shard = {"images": [], "instances": [], "negatives": []}
    started = time.time()
    acquisition_dir = data_root / ACQUISITION_DIR_NAME

    for position, target in enumerate(targets, start=1):
        image_record, instances, negatives, image_stats = label_one_image(
            predictor,
            acquisition_dir / target["path"],
            target
        )
        stats.update(image_stats)
        shard["images"].append(image_record)
        shard["instances"].extend(instances)
        shard["negatives"].extend(negatives)

        if position % SHARD_SIZE == 0 or position == len(targets):
            write_shard(shard_dir, shard)
            shard = {"images": [], "instances": [], "negatives": []}
            rate = position / (time.time() - started) * SECONDS_PER_HOUR
            logging.warning("labelled %d of %d, %.0f images/hour", position, len(targets), rate)

    return stats, len(targets), time.time() - started

def load_overlay_tile(
        image_path: Path,
        overlays: list[tuple[str, list[list[float]]]]
    ) -> np.ndarray:
    pixels, _ = load_model_pixels(image_path)
    height, width = pixels.shape[:2]
    colors = {PROMPT_FOOD: FOOD_COLOR, PROMPT_SAUCE: SAUCE_COLOR}

    for prompt, polygons in overlays:
        for polygon in polygons:
            points = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
            thickness = max(OUTLINE_THICKNESS, width // 300)
            cv2.polylines(pixels, [points], True, colors[prompt], thickness)

    scale = TILE_SIZE / max(height, width)
    resized_size = (round(width * scale), round(height * scale))
    resized = cv2.resize(pixels, resized_size, interpolation=cv2.INTER_AREA)
    tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    tile[: resized.shape[0], : resized.shape[1]] = resized

    return tile

def write_check_sheet(data_root: Path, stamp: str) -> Path:
    store_dir = data_root / STORE_DIR_NAME
    payload = json.loads((store_dir / INSTANCES_NAME).read_text(encoding="utf-8"))
    by_image = {}

    for annotation in payload["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)

    labelled = [record for record in payload["images"] if record["n_food_masks"] > 0]
    sample = random.Random(SHEET_SEED).sample(labelled, min(SAMPLE_COUNT, len(labelled)))
    rows = (len(sample) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    sheet = np.zeros((rows * TILE_SIZE, SHEET_COLUMNS * TILE_SIZE, 3), dtype=np.uint8)

    for position, record in enumerate(sample):
        overlays = []

        for item in by_image[record["image_id"]]:
            overlays.append((item["prompt"], item["polygons"]))

        image_path = data_root / ACQUISITION_DIR_NAME / record["path"]
        tile = load_overlay_tile(image_path, overlays)
        top = (position // SHEET_COLUMNS) * TILE_SIZE
        left = (position % SHEET_COLUMNS) * TILE_SIZE
        sheet[top:top + TILE_SIZE, left:left + TILE_SIZE] = tile

    sheet_path = store_dir / f"sam3_check_{stamp}.jpg"
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, SHEET_QUALITY])

    return sheet_path

def summarize_labels(data_root: Path) -> list[str]:
    images = json.loads((data_root / STORE_DIR_NAME / IMAGES_NAME).read_text(encoding="utf-8"))
    frame = pl.DataFrame(images)
    lines = ["per class: images / no_food_found / share / mean food masks"]

    for source_id in sorted(frame["source_id"].unique().to_list()):
        folder = frame.filter(pl.col("source_id") == source_id)
        empty = int(folder["no_food_found"].sum())
        share = empty / folder.height
        mean_masks = folder["n_food_masks"].mean()
        lines.append(f"{source_id}\t{folder.height}\t{empty}\t{share:.3f}\t{mean_masks:.2f}")

    return lines

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye label sam3")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    source_ids = arguments.source_id or list(PILOT_SOURCE_IDS)
    stamp = datetime.now(UTC).strftime(DATE_FORMAT)
    config = {"prompts": PROMPTS, "predictor_conf": PREDICTOR_CONF, "source_ids": source_ids}
    start_run("label-sam3", config)
    stats, label_count, seconds = run_labelling(arguments.data_root, source_ids, arguments.limit)
    store_dir = arguments.data_root / STORE_DIR_NAME
    totals = merge_shards(store_dir / SHARD_DIR_NAME, store_dir)
    sheet_path = write_check_sheet(arguments.data_root, stamp)
    lines = [f"labelled {label_count} images in {seconds:.0f} s", f"store totals {totals}"]
    lines.append(f"filter drops {dict(stats)}")

    return lines + summarize_labels(arguments.data_root) + [f"check sheet {sheet_path}"]
