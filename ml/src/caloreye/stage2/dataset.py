import random
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from caloreye.data.clean import find_latest_flags
from caloreye.data.export_s1 import DATASETS_DIR_NAME
from caloreye.data.export_s2 import (
    CLASS_SOURCE_IDS,
    CROPS_NAME,
    ORIGIN_WHOLE,
    STAGE_TWO_DIR_NAME,
)
from caloreye.data.manifest import (
    ACQUISITION_DIR_NAME,
    LATEST_MANIFEST_NAME,
    WORK_DIR_NAME,
    convert_to_rgb,
)
from caloreye.export.goldens import crop_box_with_margin
from caloreye.export.parity import prepare_stage_two_input
from caloreye.taxonomy import PILOT_SOURCE_IDS

CACHE_DIR_NAME = "s2cache"
CACHE_MAX_SIDE = 512
CACHE_JPEG_QUALITY = 95
WHOLE_SCALE_RANGE = (0.35, 1.0)
WHOLE_RATIO_RANGE = (0.75, 1.3333)
BOX_JITTER = 0.10
FLIP_PROBABILITY = 0.5
ROTATION_PROBABILITY = 0.5
ROTATION_DEGREES = 15.0
JPEG_PROBABILITY = 0.5
JPEG_QUALITY_RANGE = (40, 90)
COLOR_PROBABILITY = 0.8
COLOR_STRENGTH = 0.2
BORDER_VALUE = 128
UNSEEN_PER_FOLDER = 30
UNSEEN_SEED = 0
LOOKALIKE_SOURCE_IDS = ("7", "8", "54")
UNSEEN_DISH_INDEX = -1
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
UNCLEANED_ORIGIN = ORIGIN_WHOLE
FULL_BOX = (0.0, 0.0, 1.0, 1.0)
MIN_CROP_PIXELS = 8
MAX_CROP_ATTEMPTS = 10
SQUARE_ROOT_EXPONENT = 0.5
BYTE_MAX = 255

def cache_path_for(data_root: Path, image_id: str) -> Path:
    return data_root / WORK_DIR_NAME / CACHE_DIR_NAME / f"{image_id}.jpg"

def build_cache(data_root: Path, rows: list[dict]) -> int:
    created = 0
    seen = set()

    for row in rows:
        if row["image_id"] in seen:
            continue

        seen.add(row["image_id"])
        target = cache_path_for(data_root, row["image_id"])

        if target.is_file():
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        source = data_root / ACQUISITION_DIR_NAME / row["path"]

        with Image.open(source) as image:
            image.draft("RGB", (CACHE_MAX_SIDE * 2, CACHE_MAX_SIDE * 2))
            rgb_image = convert_to_rgb(ImageOps.exif_transpose(image))
            rgb_image.thumbnail((CACHE_MAX_SIDE, CACHE_MAX_SIDE), Image.LANCZOS)
            rgb_image.save(target, quality=CACHE_JPEG_QUALITY)

        created += 1

    return created

def load_split_rows(data_root: Path, split_name: str) -> list[dict]:
    crops_path = data_root / DATASETS_DIR_NAME / STAGE_TWO_DIR_NAME / CROPS_NAME
    frame = pl.read_parquet(crops_path).filter(pl.col("split") == split_name)

    return frame.to_dicts()

def build_uncleaned_rows(data_root: Path) -> list[dict]:
    flags = pl.read_parquet(find_latest_flags(data_root / WORK_DIR_NAME))
    manifest = pl.read_parquet(data_root / WORK_DIR_NAME / LATEST_MANIFEST_NAME)
    clusters = manifest.select(["path", "dup_cluster"])
    dropped = flags.filter(~pl.col("keep") & pl.col("source_id").is_in(list(PILOT_SOURCE_IDS)))
    dropped = dropped.filter(pl.col("is_tiny") | pl.col("is_pilot_outlier"))
    dropped = dropped.join(clusters, on="path", how="left")
    kept = flags.filter(pl.col("keep")).join(clusters, on="path", how="left")
    crops_path = data_root / DATASETS_DIR_NAME / STAGE_TWO_DIR_NAME / CROPS_NAME
    val_crops = pl.read_parquet(crops_path).filter(pl.col("split") == SPLIT_VAL)
    val_ids = val_crops["image_id"].unique()
    val_clusters = kept.filter(pl.col("image_id").is_in(val_ids))["dup_cluster"].unique()
    dropped = dropped.filter(~pl.col("dup_cluster").is_in(val_clusters))
    rows = []

    for record in dropped.iter_rows(named=True):
        dish_idx = CLASS_SOURCE_IDS.index(record["source_id"])
        rows.append(
            {
                "image_id": record["image_id"],
                "path": record["path"],
                "source_id": record["source_id"],
                "x0": FULL_BOX[0],
                "y0": FULL_BOX[1],
                "x1": FULL_BOX[2],
                "y1": FULL_BOX[3],
                "dish_idx": dish_idx,
                "origin": UNCLEANED_ORIGIN,
                "split": SPLIT_TRAIN
            }
        )

    return rows

def build_unseen_rows(data_root: Path) -> list[dict]:
    flags = pl.read_parquet(find_latest_flags(data_root / WORK_DIR_NAME)).filter(pl.col("keep"))
    rng = random.Random(UNSEEN_SEED)
    rows = []

    for source_id in sorted(flags["source_id"].unique().to_list()):
        if source_id in PILOT_SOURCE_IDS:
            continue

        folder = flags.filter(pl.col("source_id") == source_id).sort("path").to_dicts()
        limit = UNSEEN_PER_FOLDER

        if source_id in LOOKALIKE_SOURCE_IDS:
            limit = len(folder)

        for record in rng.sample(folder, min(limit, len(folder))):
            rows.append(
                {
                    "image_id": record["image_id"],
                    "path": record["path"],
                    "source_id": source_id,
                    "x0": FULL_BOX[0],
                    "y0": FULL_BOX[1],
                    "x1": FULL_BOX[2],
                    "y1": FULL_BOX[3],
                    "dish_idx": UNSEEN_DISH_INDEX,
                    "origin": ORIGIN_WHOLE,
                    "split": "unseen"
                }
            )

    return rows

def pixel_box(row: dict, width: int, height: int) -> tuple[float, float, float, float]:
    return (row["x0"] * width, row["y0"] * height, row["x1"] * width, row["y1"] * height)

def jitter_box(box: tuple, rng: random.Random) -> tuple[float, float, float, float]:
    left, top, right, bottom = box
    box_width = right - left
    box_height = bottom - top

    return (
        left + rng.uniform(-BOX_JITTER, BOX_JITTER) * box_width,
        top + rng.uniform(-BOX_JITTER, BOX_JITTER) * box_height,
        right + rng.uniform(-BOX_JITTER, BOX_JITTER) * box_width,
        bottom + rng.uniform(-BOX_JITTER, BOX_JITTER) * box_height
    )

def crop_with_margin(image: np.ndarray, box: tuple) -> np.ndarray:
    height, width = image.shape[:2]
    left, top, right, bottom = crop_box_with_margin(box, width, height)

    if right - left < MIN_CROP_PIXELS or bottom - top < MIN_CROP_PIXELS:
        return image

    return image[top:bottom, left:right]

def random_resized_crop(image: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = image.shape[:2]
    area = height * width
    attempts = 0

    while attempts < MAX_CROP_ATTEMPTS:
        attempts += 1
        target_area = area * rng.uniform(*WHOLE_SCALE_RANGE)
        ratio = rng.uniform(*WHOLE_RATIO_RANGE)
        crop_width = int(round((target_area * ratio) ** SQUARE_ROOT_EXPONENT))
        crop_height = int(round((target_area / ratio) ** SQUARE_ROOT_EXPONENT))

        if 0 < crop_width <= width and 0 < crop_height <= height:
            left = rng.randint(0, width - crop_width)
            top = rng.randint(0, height - crop_height)

            return image[top:top + crop_height, left:left + crop_width]

    return image

def rotate_image(image: np.ndarray, degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderValue=(BORDER_VALUE, BORDER_VALUE, BORDER_VALUE)
    )

def jitter_colors(image: np.ndarray, rng: random.Random) -> np.ndarray:
    brightness = 1.0 + rng.uniform(-COLOR_STRENGTH, COLOR_STRENGTH)
    contrast = 1.0 + rng.uniform(-COLOR_STRENGTH, COLOR_STRENGTH)
    saturation = 1.0 + rng.uniform(-COLOR_STRENGTH, COLOR_STRENGTH)
    working = image.astype(np.float32) * brightness
    mean_value = working.mean()
    working = (working - mean_value) * contrast + mean_value
    gray = working.mean(axis=2, keepdims=True)
    working = (working - gray) * saturation + gray

    return np.clip(working, 0, BYTE_MAX).astype(np.uint8)

def recompress_jpeg(image: np.ndarray, rng: random.Random) -> np.ndarray:
    quality = rng.randint(*JPEG_QUALITY_RANGE)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    if not success:
        return image

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

def augment_image(image: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < FLIP_PROBABILITY:
        image = np.ascontiguousarray(image[:, ::-1])

    if rng.random() < ROTATION_PROBABILITY:
        image = rotate_image(image, rng.uniform(-ROTATION_DEGREES, ROTATION_DEGREES))

    if rng.random() < COLOR_PROBABILITY:
        image = jitter_colors(image, rng)

    if rng.random() < JPEG_PROBABILITY:
        image = recompress_jpeg(image, rng)

    return image

def read_cached_image(data_root: Path, image_id: str) -> np.ndarray:
    with Image.open(cache_path_for(data_root, image_id)) as image:
        return np.asarray(image.convert("RGB"))

def make_crop(image: np.ndarray, row: dict, is_train: bool, rng: random.Random) -> np.ndarray:
    height, width = image.shape[:2]

    if row["origin"] == ORIGIN_WHOLE:
        if is_train:
            return random_resized_crop(image, rng)

        return image

    box = pixel_box(row, width, height)

    if is_train:
        box = jitter_box(box, rng)

    return crop_with_margin(image, box)

class CropDataset(Dataset):
    def __init__(self, data_root: Path, rows: list[dict], is_train: bool, seed: int):
        self.data_root = data_root
        self.rows = rows
        self.is_train = is_train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.rows[index]
        rng = random.Random()

        if not self.is_train:
            rng = random.Random(self.seed + index)

        image = read_cached_image(self.data_root, row["image_id"])
        crop = make_crop(image, row, self.is_train, rng)

        if self.is_train:
            crop = augment_image(crop, rng)

        tensor = prepare_stage_two_input(np.ascontiguousarray(crop))[0]

        return tensor, int(row["dish_idx"]), index

def sampler_weights(rows: list[dict], class_count: int) -> list[float]:
    counts = [0] * class_count

    for row in rows:
        counts[row["dish_idx"]] += 1

    return [1.0 / counts[row["dish_idx"]] for row in rows]

