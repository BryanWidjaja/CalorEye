import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import polars as pl
from PIL import Image

from caloreye.data.dedupe import cluster_by_hash
from caloreye.paths import get_data_root_dir
from caloreye.runs import hash_file, start_run

ACQUISITION_DIR_NAME = "acquisition"
IMAGES_DIR_NAME = "images"
CANDIDATES_NAME = "candidates.jsonl"
WORK_DIR_NAME = "work"
LATEST_MANIFEST_NAME = "manifest.parquet"
RESULT_NAME = "manifest_result.json"
DATE_FORMAT = "%Y-%m-%d"
PHASH_SIZE = 32
PHASH_LOW_BAND = 8
HASH_CHUNK_BYTES = 1024 * 1024
DRAFT_SIZE = (256, 256)
WHITE_FILL = (255, 255, 255, 255)
SIXTEEN_BIT_SCALE = 257
BYTE_MAX = 255
TOP_CROSS_FOLDER_COUNT = 20
PROGRESS_INTERVAL = 2000
JSON_INDENT = 2
UNDECODABLE_MESSAGE = "undecodable image %s: %s"
SECOND_LEVEL_LABELS = ("co", "com", "or", "go", "ac", "web", "net", "org", "edu", "gov")
COUNTRY_CODE_LENGTH = 2
MIN_LABELS_FOR_SECOND_LEVEL = 3
SIXTEEN_BIT_MODES = ("I;16", "I;16L", "I;16B", "I")
ALPHA_MODES = ("P", "PA", "LA", "RGBA", "La")
HOST_RECORD_KEYS = ("file_page_url", "original_image_url", "uploader_group")
SCHEMA = {
    "image_id": pl.String,
    "path": pl.String,
    "source_id": pl.String,
    "width": pl.Int32,
    "height": pl.Int32,
    "bytes": pl.Int64,
    "mime": pl.String,
    "host": pl.String,
    "page_url": pl.String,
    "original_sha256": pl.String,
    "is_derivative": pl.Boolean,
    "decodable": pl.Boolean,
    "phash": pl.UInt64
}

@dataclass
class ImageFacts:
    width: int
    height: int
    phash: int

def registered_domain(url_or_host: str) -> str:
    if not url_or_host:
        return ""

    target = url_or_host

    if "//" not in target:
        target = f"//{target}"

    host = (urlparse(target).hostname or "").lower()
    labels = host.split(".")

    if len(labels) <= COUNTRY_CODE_LENGTH:
        return host

    has_second_level = labels[-2] in SECOND_LEVEL_LABELS
    has_country_code = len(labels[-1]) == COUNTRY_CODE_LENGTH

    if has_second_level and has_country_code and len(labels) >= MIN_LABELS_FOR_SECOND_LEVEL:
        return ".".join(labels[-3:])

    return ".".join(labels[-2:])

def convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image

    if image.mode in SIXTEEN_BIT_MODES:
        scaled = np.asarray(image, dtype=np.uint32) // SIXTEEN_BIT_SCALE
        gray_array = np.clip(scaled, 0, BYTE_MAX).astype(np.uint8)

        return Image.fromarray(gray_array, mode="L").convert("RGB")

    if image.mode in ALPHA_MODES:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, WHITE_FILL)
        background.alpha_composite(rgba)

        return background.convert("RGB")

    return image.convert("RGB")

def compute_phash(rgb_array: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (PHASH_SIZE, PHASH_SIZE), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(small.astype(np.float32))
    low_band = coefficients[:PHASH_LOW_BAND, :PHASH_LOW_BAND].flatten()
    bits = low_band > np.median(low_band)
    value = 0

    for bit in bits:
        value = (value << 1) | int(bit)

    return value

def read_image_facts(image_path: Path) -> ImageFacts:
    with Image.open(image_path) as image:
        width, height = image.size
        image.draft("RGB", DRAFT_SIZE)
        rgb_array = np.asarray(convert_to_rgb(image))

    return ImageFacts(width=width, height=height, phash=compute_phash(rgb_array))

def hash_bytes(image_path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0

    with image_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
            size_bytes += len(chunk)

    return digest.hexdigest(), size_bytes

def load_candidates(candidates_path: Path) -> dict[tuple[str, str], dict]:
    records = {}

    if not candidates_path.is_file():
        return records

    with candidates_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            record = json.loads(line)
            file_name = Path(record.get("local_path") or "").name
            records[(record["source_id"], file_name)] = record

    return records

def resolve_host(record: dict) -> str:
    for key in HOST_RECORD_KEYS:
        domain = registered_domain(record.get(key) or "")

        if domain:
            return domain

    return ""

def build_row(image_path: Path, acquisition_dir: Path, record: dict) -> dict:
    image_id, size_bytes = hash_bytes(image_path)
    original_sha256 = record.get("original_sha256") or ""
    row = {
        "image_id": image_id,
        "path": image_path.relative_to(acquisition_dir).as_posix(),
        "source_id": image_path.parent.name,
        "width": None,
        "height": None,
        "bytes": size_bytes,
        "mime": record.get("mime") or "",
        "host": resolve_host(record),
        "page_url": record.get("file_page_url") or "",
        "original_sha256": original_sha256,
        "is_derivative": None,
        "decodable": False,
        "phash": 0
    }

    if original_sha256:
        row["is_derivative"] = original_sha256 != image_id

    try:
        facts = read_image_facts(image_path)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        logging.warning(UNDECODABLE_MESSAGE, image_path.name, error)

        return row

    row["width"] = facts.width
    row["height"] = facts.height
    row["phash"] = facts.phash
    row["decodable"] = True

    return row

def list_image_paths(images_dir: Path) -> list[Path]:
    image_paths = []

    for source_dir in sorted(images_dir.iterdir()):
        if not source_dir.is_dir():
            continue

        for image_path in sorted(source_dir.iterdir()):
            if image_path.is_file():
                image_paths.append(image_path)

    return image_paths

def scan_rows(acquisition_dir: Path, candidates: dict) -> list[dict]:
    rows = []
    image_paths = list_image_paths(acquisition_dir / IMAGES_DIR_NAME)

    if candidates:
        image_paths = [path for path in image_paths if (path.parent.name, path.name) in candidates]

    for position, image_path in enumerate(image_paths):
        record = candidates.get((image_path.parent.name, image_path.name), {})
        rows.append(build_row(image_path, acquisition_dir, record))

        if (position + 1) % PROGRESS_INTERVAL == 0:
            logging.warning("scanned %d of %d", position + 1, len(image_paths))

    return rows

def assign_clusters(frame: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    decodable = frame.filter(pl.col("decodable"))
    image_ids = decodable["image_id"].to_list()
    result = cluster_by_hash(image_ids, decodable["phash"].to_numpy())
    cluster_frame = pl.DataFrame(
        {
            "image_id": image_ids,
            "dup_cluster": result.cluster_ids
        },
        schema={"image_id": pl.String, "dup_cluster": pl.Int64}
    ).unique(subset=["image_id"])
    joined = frame.join(cluster_frame, on="image_id", how="left")
    sizes = joined.group_by("dup_cluster").agg(pl.len().cast(pl.Int32).alias("cluster_size"))

    return joined.join(sizes, on="dup_cluster", how="left"), result.near_pair_count

def summarize(frame: pl.DataFrame, near_pair_count: int) -> list[str]:
    total = frame.height
    undecodable = frame.filter(~pl.col("decodable")).height
    unique_ids = frame["image_id"].n_unique()
    clustered = frame.filter(pl.col("cluster_size") > 1)
    lines = [
        f"files {total}  undecodable {undecodable}  unique byte hashes {unique_ids}",
        f"exact duplicate files {total - unique_ids}",
        f"near duplicate pairs (hamming<=4, unique images) {near_pair_count}",
        f"clusters with more than one file {clustered['dup_cluster'].n_unique()}"
        f"  largest {frame['cluster_size'].max()}",
        "",
        "files per folder"
    ]

    for source_id, count in frame.group_by("source_id").len().sort("source_id").iter_rows():
        lines.append(f"{source_id}\t{count}")

    return lines

def cross_folder_lines(frame: pl.DataFrame) -> list[str]:
    spans = (
        frame.filter(pl.col("cluster_size") > 1)
        .group_by("dup_cluster")
        .agg(
            pl.col("source_id").n_unique().alias("folder_count"),
            pl.col("source_id").unique().sort().alias("folders"),
            pl.len().alias("files")
        )
        .filter(pl.col("folder_count") > 1)
        .sort(["files", "folder_count"], descending=True)
    )
    lines = [f"clusters spanning more than one folder {spans.height}", ""]

    for record in spans.head(TOP_CROSS_FOLDER_COUNT).iter_rows(named=True):
        folders = ",".join(record["folders"])
        lines.append(f"cluster {record['dup_cluster']}\tfiles {record['files']}\tfolders {folders}")

    return lines

def write_manifest(frame: pl.DataFrame, work_dir: Path, stamp: str) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    dated_path = work_dir / f"manifest-{stamp}.parquet"
    dated_temp = dated_path.with_suffix(".tmp")
    latest_path = work_dir / LATEST_MANIFEST_NAME
    latest_temp = latest_path.with_suffix(".tmp")
    frame.write_parquet(dated_temp)
    frame.write_parquet(latest_temp)
    dated_temp.replace(dated_path)
    latest_temp.replace(latest_path)

    return dated_path

def attach_triage_flags(frame: pl.DataFrame, data_root: Path) -> pl.DataFrame:
    acquisition_dir = data_root / ACQUISITION_DIR_NAME
    marker_path = acquisition_dir / "clean" / "applied.json"

    if not marker_path.is_file():
        return frame

    decision_path = acquisition_dir / "clean" / "decisions.parquet"

    if not decision_path.is_file():
        raise FileNotFoundError(f"triage applied but decisions missing: {decision_path}")

    decisions = pl.read_parquet(decision_path)
    paths = []

    for local_path in decisions["local_path"]:
        paths.append(Path(local_path).relative_to(acquisition_dir).as_posix())

    flags = decisions.select(
        pl.Series("path", paths),
        pl.col("decision").alias("triage_decision"),
        pl.col("reasons").alias("triage_reasons")
    )
    joined = frame.join(flags, on="path", how="left")

    if joined["triage_decision"].null_count():
        raise ValueError("triage decisions do not cover every manifest image")

    return joined

def build_manifest(data_root: Path, stamp: str) -> tuple[pl.DataFrame, list[str], Path]:
    acquisition_dir = data_root / ACQUISITION_DIR_NAME
    candidates = load_candidates(acquisition_dir / CANDIDATES_NAME)
    rows = scan_rows(acquisition_dir, candidates)
    frame = pl.DataFrame(rows, schema=SCHEMA, strict=False)
    frame, near_pair_count = assign_clusters(frame)
    frame = attach_triage_flags(frame, data_root)
    manifest_path = write_manifest(frame, data_root / WORK_DIR_NAME, stamp)
    report = summarize(frame, near_pair_count) + [""] + cross_folder_lines(frame)

    return frame, report, manifest_path

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye data manifest")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    arguments = parser.parse_args(argv)
    stamp = datetime.now(UTC).strftime(DATE_FORMAT)
    frame, report, manifest_path = build_manifest(arguments.data_root, stamp)
    run_dir = start_run("data-manifest", {"data_root": str(arguments.data_root)}).run_dir
    manifest_hash = hash_file(manifest_path)
    report_path = arguments.data_root / WORK_DIR_NAME / f"manifest-report-{stamp}.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    result = {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "rows": frame.height
    }
    result_text = json.dumps(result, indent=JSON_INDENT) + "\n"
    (run_dir / RESULT_NAME).write_text(result_text, encoding="utf-8")

    return report + ["", f"manifest {manifest_path}", f"sha256 {manifest_hash}", f"report {report_path}"]
