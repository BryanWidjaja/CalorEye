import argparse
import hashlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from PIL import Image, ImageOps, UnidentifiedImageError

from caloreye.data.manifest import compute_phash, convert_to_rgb
from caloreye.paths import get_data_root_dir
from caloreye.runs import start_run

CHUNK_SIZE = 2048
PROGRESS_INTERVAL = 5000
SHORT_SIDE_MIN = 160
ASPECT_MAX = 4.0
LUMA_STD_MIN = 8.0
SHARPNESS_SIZE = 256
HASH_SIZE = 9
EDGE_LOW = 35
EDGE_HIGH = 110
CORNER_FRACTION = 0.22
TEXT_BAND_FRACTION = 0.2
PROBE_VERSION = "1"
DECODE_ERRORS = (OSError, ValueError, UnidentifiedImageError, cv2.error)

def score_pixels(rgb_array: np.ndarray) -> dict:
    height, width = rgb_array.shape[:2]
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (SHARPNESS_SIZE, SHARPNESS_SIZE), interpolation=cv2.INTER_AREA)
    channels = rgb_array.astype(np.float32)
    red_green = np.abs(channels[:, :, 0] - channels[:, :, 1])
    yellow_blue = np.abs((channels[:, :, 0] + channels[:, :, 1]) / 2 - channels[:, :, 2])
    edges = cv2.Canny(resized, EDGE_LOW, EDGE_HIGH)
    band_width = max(1, round(SHARPNESS_SIZE * TEXT_BAND_FRACTION))
    corner_width = max(1, round(SHARPNESS_SIZE * CORNER_FRACTION))
    bands = np.concatenate((edges[:band_width].ravel(), edges[-band_width:].ravel()))
    corner = np.concatenate((edges[:corner_width, :corner_width].ravel(), edges[-corner_width:, -corner_width:].ravel()))
    hash_pixels = cv2.resize(gray, (HASH_SIZE, HASH_SIZE - 1), interpolation=cv2.INTER_AREA)
    bits = hash_pixels[:, 1:] > hash_pixels[:, :-1]
    dhash = 0

    for bit in bits.ravel():
        dhash = (dhash << 1) | int(bit)

    return {
        "mean_luma": float(gray.mean()),
        "luma_std": float(gray.std()),
        "colorfulness": float(np.sqrt(red_green.std() ** 2 + yellow_blue.std() ** 2)),
        "sharpness": float(cv2.Laplacian(resized, cv2.CV_64F).var()),
        "text_density": float(np.count_nonzero(bands) / bands.size),
        "corner_overlay_score": float(np.count_nonzero(corner) / corner.size),
        "dhash": f"{dhash:016x}",
        "phash": f"{compute_phash(rgb_array):016x}",
        "width": width,
        "height": height
    }

def hard_reasons(facts: dict) -> list[str]:
    reasons = []

    if not facts["decodes"]:
        reasons.append("undecodable")
        return reasons

    if facts["short_side"] < SHORT_SIDE_MIN:
        reasons.append("tiny")

    if facts["aspect"] > ASPECT_MAX:
        reasons.append("extreme_aspect")

    if facts["luma_std"] < LUMA_STD_MIN:
        reasons.append("blank")

    return reasons

def probe_candidate(candidate: dict) -> dict:
    image_path = Path(candidate["local_path"])
    facts = {
        "candidate_id": candidate["candidate_id"],
        "source_id": candidate["source_id"],
        "query_id": candidate.get("query_id", ""),
        "discovery_route": candidate.get("discovery_route", ""),
        "local_path": str(image_path),
        "original_sha256": candidate.get("original_sha256", ""),
        "decodes": False,
        "file_bytes": 0,
        "width": 0,
        "height": 0,
        "short_side": 0,
        "aspect": 0.0,
        "mean_luma": 0.0,
        "luma_std": 0.0,
        "colorfulness": 0.0,
        "sharpness": 0.0,
        "text_density": 0.0,
        "corner_overlay_score": 0.0,
        "dhash": "",
        "phash": "",
        "version": PROBE_VERSION
    }

    try:
        facts["file_bytes"] = image_path.stat().st_size

        with Image.open(image_path) as image:
            image.seek(0)
            upright = ImageOps.exif_transpose(image)
            rgb_array = np.asarray(convert_to_rgb(upright))

        facts.update(score_pixels(rgb_array))
        facts["decodes"] = True
        facts["short_side"] = min(facts["width"], facts["height"])
        facts["aspect"] = max(facts["width"], facts["height"]) / facts["short_side"]
    except DECODE_ERRORS:
        pass

    facts["hard_reasons"] = hard_reasons(facts)
    return facts

def read_candidate_chunks(candidate_path: Path):
    with candidate_path.open(encoding="utf-8") as candidates:
        chunk = []

        for line in candidates:
            chunk.append(json.loads(line))

            if len(chunk) == CHUNK_SIZE:
                yield chunk
                chunk = []

        if chunk:
            yield chunk

def part_path_for(chunk: list[dict], parts_dir: Path) -> Path:
    digest = hashlib.sha256()

    for candidate in chunk:
        digest.update(candidate["candidate_id"].encode())

    return parts_dir / f"{digest.hexdigest()[:20]}.parquet"

def write_part(facts: list[dict], part_path: Path) -> None:
    temporary_path = part_path.with_suffix(".tmp")
    pl.DataFrame(facts, schema_overrides={"hard_reasons": pl.List(pl.String)}).write_parquet(temporary_path)
    temporary_path.replace(part_path)

def mark_exact_duplicates(frame: pl.LazyFrame) -> pl.LazyFrame:
    ranked = frame.with_columns((pl.col("width") * pl.col("height")).alias("pixel_count"))
    ranked = ranked.with_columns(
        pl.when(pl.col("original_sha256") != "")
        .then(pl.col("pixel_count").rank("ordinal", descending=True).over("original_sha256"))
        .otherwise(1)
        .alias("sha_rank")
    )
    ranked = ranked.with_columns(
        pl.when(pl.col("sha_rank") > 1)
        .then(pl.concat_list(pl.col("hard_reasons"), pl.lit("exact_duplicate")))
        .otherwise(pl.col("hard_reasons"))
        .alias("hard_reasons")
    )
    return ranked.drop("pixel_count", "sha_rank")

def run_probe(data_root: Path, workers: int) -> tuple[Path, int]:
    candidate_path = data_root / "acquisition" / "candidates.jsonl"

    if not candidate_path.is_file():
        raise FileNotFoundError(f"missing candidate manifest: {candidate_path}")

    clean_dir = data_root / "acquisition" / "clean"
    parts_dir = clean_dir / "probe_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_root / "triage_probe.log"
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(message)s", force=True)
    start_run("data-triage-probe", {"version": PROBE_VERSION, "workers": workers})
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for chunk in read_candidate_chunks(candidate_path):
            part_path = part_path_for(chunk, parts_dir)

            if not part_path.is_file():
                facts = list(executor.map(probe_candidate, chunk, chunksize=32))
                write_part(facts, part_path)

            completed += len(chunk)

            if completed // PROGRESS_INTERVAL > (completed - len(chunk)) // PROGRESS_INTERVAL:
                logging.info("probed %s images", completed)

    part_paths = sorted(parts_dir.glob("*.parquet"))
    output_path = clean_dir / "probe.parquet"
    temporary_path = output_path.with_suffix(".tmp")
    parts = []

    for part_path in part_paths:
        part = pl.scan_parquet(part_path).with_columns(
            pl.col("hard_reasons").cast(pl.List(pl.String))
        )
        parts.append(part)

    frame = pl.concat(parts).unique(subset=["candidate_id"], keep="last").sort("candidate_id")
    mark_exact_duplicates(frame).sink_parquet(temporary_path)
    temporary_path.replace(output_path)
    return output_path, completed

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye data triage")
    parser.add_argument("step", choices=["probe"])
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2)))
    arguments = parser.parse_args(argv)

    if arguments.workers < 1 or arguments.workers > max(1, (os.cpu_count() or 2) - 2):
        parser.error("workers must be between 1 and cpu_count minus 2")

    output_path, count = run_probe(arguments.data_root, arguments.workers)
    return [f"probed {count} candidates", f"probe {output_path}"]
