import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image, ImageDraw, ImageOps

from caloreye.data.embed import embedding_dir_for, load_embeddings
from caloreye.data.manifest import (
    ACQUISITION_DIR_NAME,
    DATE_FORMAT,
    LATEST_MANIFEST_NAME,
    WORK_DIR_NAME,
    convert_to_rgb,
)
from caloreye.paths import get_data_root_dir
from caloreye.runs import hash_file, start_run
from caloreye.taxonomy import PILOT_SOURCE_IDS

NEIGHBOR_COUNT = 10
MIN_FOLDER_SIZE = 3
CENTROID_ITERATIONS = 4
CENTROID_KEEP_FRACTION = 0.7
MAD_SCALE = 1.4826
OUTLIER_Z_LIMIT = 3.5
TINY_SIDE_PIXELS = 256
SHEET_WORST_COUNT = 100
SHEET_COLUMNS = 6
SHEET_ROWS = 5
TILE_SIZE = 200
CAPTION_HEIGHT = 16
KNN_CHUNK = 4096
EPSILON = 1e-9
SHEET_DIR_NAME = "clean_sheets"
REVIEW_TEMPLATE_NAME = "review_template.csv"
CLEAN_PREFIX = "clean_flags"
REVIEW_COLUMNS = ["image_id", "decision", "reason"]
KEEP_DECISION = "keep"
DROP_DECISION = "drop"
SHEET_BACKGROUND = (30, 30, 30)
CAPTION_COLOR = (255, 255, 0)
SHEET_QUALITY = 85
DRAFT_SIZE = (TILE_SIZE * 2, TILE_SIZE * 2)

def compute_outlier_scores(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) < MIN_FOLDER_SIZE:
        return np.zeros(len(vectors), dtype=np.float32)

    centroid = np.median(vectors, axis=0)
    centroid /= np.linalg.norm(centroid) + EPSILON
    remaining = CENTROID_ITERATIONS

    while remaining > 0:
        distances = 1.0 - vectors @ centroid
        cutoff = np.quantile(distances, CENTROID_KEEP_FRACTION)
        member_mask = distances <= cutoff
        centroid = np.median(vectors[member_mask], axis=0)
        centroid /= np.linalg.norm(centroid) + EPSILON
        remaining -= 1

    return (1.0 - vectors @ centroid).astype(np.float32)

def robust_z_scores(scores: np.ndarray) -> np.ndarray:
    center = np.median(scores)
    spread = MAD_SCALE * np.median(np.abs(scores - center))

    return ((scores - center) / (spread + EPSILON)).astype(np.float32)

def compute_folder_outliers(
        vectors: np.ndarray,
        folder_codes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros(len(vectors), dtype=np.float32)
    z_scores = np.zeros(len(vectors), dtype=np.float32)

    for code in np.unique(folder_codes):
        positions = np.flatnonzero(folder_codes == code)
        folder_scores = compute_outlier_scores(vectors[positions].astype(np.float32))
        scores[positions] = folder_scores
        z_scores[positions] = robust_z_scores(folder_scores)

    return scores, z_scores

def compute_knn_conflict(
        vectors: np.ndarray,
        folder_codes: np.ndarray,
        cluster_ids: np.ndarray
    ) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = torch.from_numpy(vectors).to(device)
    folders = torch.from_numpy(folder_codes).to(device)
    clusters = torch.from_numpy(cluster_ids).to(device)
    conflicts = np.zeros(len(vectors), dtype=np.float32)

    for start in range(0, len(vectors), KNN_CHUNK):
        end = min(start + KNN_CHUNK, len(vectors))
        similarity = (matrix[start:end] @ matrix.T).float()
        same_cluster = clusters[start:end, None] == clusters[None, :]
        similarity[same_cluster] = -2.0
        neighbors = similarity.topk(NEIGHBOR_COUNT, dim=1).indices
        different = folders[neighbors] != folders[start:end, None]
        conflicts[start:end] = different.float().mean(dim=1).cpu().numpy()

    return conflicts

def choose_keepers(frame: pl.DataFrame) -> pl.DataFrame:
    ranked = frame.with_columns(
        (pl.col("width").cast(pl.Int64) * pl.col("height").cast(pl.Int64)).alias("pixel_count")
    ).sort(
        ["dup_cluster", "source_id", "pixel_count", "image_id"],
        descending=[False, False, True, False]
    )
    ranked = ranked.with_columns(
        pl.int_range(pl.len()).over(["dup_cluster", "source_id"]).alias("rank_in_group")
    )

    return ranked.with_columns((pl.col("rank_in_group") > 0).alias("is_duplicate_extra")).drop(
        ["pixel_count", "rank_in_group"]
    )

def build_flags(data_root: Path) -> pl.DataFrame:
    manifest = pl.read_parquet(data_root / WORK_DIR_NAME / LATEST_MANIFEST_NAME)
    manifest_hash = hash_file(data_root / WORK_DIR_NAME / LATEST_MANIFEST_NAME)
    image_ids, vectors = load_embeddings(embedding_dir_for(data_root, manifest_hash))
    index_by_id = {image_id: position for position, image_id in enumerate(image_ids)}
    decodable = manifest.filter(pl.col("decodable")).sort("path")
    row_positions = np.array([index_by_id[image_id] for image_id in decodable["image_id"]])
    row_vectors = vectors[row_positions]
    folder_names = decodable["source_id"].to_list()
    code_by_name = {name: code for code, name in enumerate(sorted(set(folder_names)))}
    folder_codes = np.array([code_by_name[name] for name in folder_names], dtype=np.int64)
    cluster_ids = decodable["dup_cluster"].to_numpy().astype(np.int64)
    scores, z_scores = compute_folder_outliers(row_vectors, folder_codes)
    conflicts = compute_knn_conflict(row_vectors, folder_codes, cluster_ids)
    flagged = decodable.with_columns(
        pl.Series("outlier_score", scores),
        pl.Series("outlier_z", z_scores),
        pl.Series("knn_conflict", conflicts)
    )

    return finish_flags(choose_keepers(flagged))

def finish_flags(frame: pl.DataFrame) -> pl.DataFrame:
    is_pilot = pl.col("source_id").is_in(list(PILOT_SOURCE_IDS))
    is_tiny = pl.min_horizontal(pl.col("width"), pl.col("height")) < TINY_SIDE_PIXELS
    is_outlier = is_pilot & (pl.col("outlier_z") > OUTLIER_Z_LIMIT)
    frame = frame.with_columns(
        is_tiny.alias("is_tiny"),
        is_outlier.alias("is_pilot_outlier")
    )
    drop_any = pl.col("is_tiny") | pl.col("is_duplicate_extra") | pl.col("is_pilot_outlier")

    return frame.with_columns((~drop_any).alias("keep")).select(
        [
            "image_id", "path", "source_id", "outlier_score", "outlier_z", "knn_conflict",
            "is_tiny", "is_duplicate_extra", "is_pilot_outlier", "keep"
        ]
    )

def removal_report(flags: pl.DataFrame) -> list[str]:
    lines = ["pilot class removals (tiny / duplicate / outlier / total dropped / kept of total)"]

    for source_id in PILOT_SOURCE_IDS:
        folder = flags.filter(pl.col("source_id") == source_id)
        dropped = folder.filter(~pl.col("keep")).height
        counts = [
            source_id,
            folder["is_tiny"].sum(),
            folder["is_duplicate_extra"].sum(),
            folder["is_pilot_outlier"].sum(),
            dropped,
            f"{folder['keep'].sum()}/{folder.height}"
        ]
        lines.append("\t".join(str(count) for count in counts))

    lines.append(f"all folders: dropped {flags.filter(~pl.col('keep')).height} of {flags.height}")

    return lines

def load_tile(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        image.draft("RGB", DRAFT_SIZE)
        upright = ImageOps.exif_transpose(image)

        return ImageOps.fit(convert_to_rgb(upright), (TILE_SIZE, TILE_SIZE))

def render_sheet(rows: list[dict], acquisition_dir: Path, output_path: Path) -> None:
    sheet = Image.new(
        "RGB",
        (SHEET_COLUMNS * TILE_SIZE, SHEET_ROWS * (TILE_SIZE + CAPTION_HEIGHT)),
        SHEET_BACKGROUND
    )
    draw = ImageDraw.Draw(sheet)

    for position, row in enumerate(rows):
        column = position % SHEET_COLUMNS
        line = position // SHEET_COLUMNS
        left = column * TILE_SIZE
        top = line * (TILE_SIZE + CAPTION_HEIGHT)
        sheet.paste(load_tile(acquisition_dir / row["path"]), (left, top + CAPTION_HEIGHT))
        caption = f"{row['rank']} z{row['outlier_z']:.1f} {row['image_id'][:8]}"
        draw.text((left + 2, top + 2), caption, fill=CAPTION_COLOR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=SHEET_QUALITY)

def write_review_material(flags: pl.DataFrame, data_root: Path, stamp: str) -> list[str]:
    acquisition_dir = data_root / ACQUISITION_DIR_NAME
    sheet_dir = data_root / WORK_DIR_NAME / SHEET_DIR_NAME / stamp
    template_rows = []
    sheet_size = SHEET_COLUMNS * SHEET_ROWS
    lines = []

    for source_id in PILOT_SOURCE_IDS:
        folder = flags.filter(pl.col("source_id") == source_id)
        worst = folder.sort("outlier_z", descending=True).head(SHEET_WORST_COUNT)
        records = worst.to_dicts()

        for rank, record in enumerate(records, start=1):
            record["rank"] = rank
            template_rows.append([record["image_id"], "", f"outlier_z={record['outlier_z']:.2f}"])

        for start in range(0, len(records), sheet_size):
            sheet_path = sheet_dir / f"{source_id}-{start // sheet_size + 1}.jpg"
            render_sheet(records[start:start + sheet_size], acquisition_dir, sheet_path)

        lines.append(f"sheets for {source_id}: {len(records)} images in {sheet_dir}")

    template_path = data_root / WORK_DIR_NAME / REVIEW_TEMPLATE_NAME
    write_review_template(template_rows, template_path)
    lines.append(f"review template {template_path}")

    return lines

def write_review_template(template_rows: list[list[str]], template_path: Path) -> None:
    with template_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(REVIEW_COLUMNS)
        writer.writerows(template_rows)

def read_review(review_path: Path) -> dict[str, str]:
    decisions = {}

    with review_path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            decision = record["decision"].strip().lower()

            if decision not in (KEEP_DECISION, DROP_DECISION):
                continue

            decisions[record["image_id"]] = decision

    return decisions

def apply_review(flags: pl.DataFrame, decisions: dict[str, str]) -> pl.DataFrame:
    lookup = pl.col("image_id").replace_strict(decisions, default=None, return_dtype=pl.String)
    reviewed = flags.with_columns(lookup.alias("review_decision"))
    final_keep = (
        pl.when(pl.col("review_decision") == KEEP_DECISION).then(True)
        .when(pl.col("review_decision") == DROP_DECISION).then(False)
        .otherwise(pl.col("keep"))
    )

    return reviewed.with_columns(final_keep.alias("keep"))

def find_latest_flags(work_dir: Path) -> Path:
    candidates = []

    for flags_path in work_dir.glob(f"{CLEAN_PREFIX}-*.parquet"):
        if not flags_path.stem.endswith("-reviewed"):
            candidates.append(flags_path)

    if not candidates:
        raise FileNotFoundError(f"expected a {CLEAN_PREFIX} file in {work_dir}, found none")

    return sorted(candidates)[-1]

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye data clean")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    parser.add_argument("--apply-review", type=Path)
    arguments = parser.parse_args(argv)
    data_root = arguments.data_root
    stamp = datetime.now(UTC).strftime(DATE_FORMAT)
    work_dir = data_root / WORK_DIR_NAME

    if arguments.apply_review is not None:
        flags_path = find_latest_flags(work_dir)
        flags = apply_review(pl.read_parquet(flags_path), read_review(arguments.apply_review))
        reviewed_path = flags_path.with_name(f"{flags_path.stem}-reviewed.parquet")
        flags.write_parquet(reviewed_path)
        start_run("data-clean-review", {"review": str(arguments.apply_review)})

        return removal_report(flags) + [f"wrote {reviewed_path}"]

    flags = build_flags(data_root)
    flags_path = work_dir / f"{CLEAN_PREFIX}-{stamp}.parquet"
    flags.write_parquet(flags_path)
    start_run("data-clean", {"flags": str(flags_path)})

    review_lines = write_review_material(flags, data_root, stamp)

    return removal_report(flags) + review_lines + [f"wrote {flags_path}"]
