import argparse
import csv
from pathlib import Path

import polars as pl
from PIL import Image, ImageDraw, ImageOps

from caloreye.data.manifest import convert_to_rgb
from caloreye.data.triage_decide import wilson_interval
from caloreye.paths import get_data_root_dir
from caloreye.taxonomy import PILOT_SOURCE_IDS

SHEET_COLUMNS = 10
SHEET_ROWS = 5
TILE_SIZE = 200
CAPTION_HEIGHT = 24
SHEET_SIZE = SHEET_COLUMNS * SHEET_ROWS
SHEET_QUALITY = 85
BACKGROUND = (25, 25, 25)
CAPTION_COLOR = (255, 240, 60)

def sheet_dimensions() -> tuple[int, int]:
    width = SHEET_COLUMNS * TILE_SIZE
    height = SHEET_ROWS * (TILE_SIZE + CAPTION_HEIGHT)
    return width, height

def create_sheet(rows: list[dict], output_path: Path, score_name: str) -> None:
    sheet = Image.new("RGB", sheet_dimensions(), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    for position, row in enumerate(rows):
        column = position % SHEET_COLUMNS
        grid_row = position // SHEET_COLUMNS
        left = column * TILE_SIZE
        top = grid_row * (TILE_SIZE + CAPTION_HEIGHT)

        with Image.open(row["local_path"]) as image:
            upright = ImageOps.exif_transpose(image)
            tile = ImageOps.pad(convert_to_rgb(upright), (TILE_SIZE, TILE_SIZE), color=BACKGROUND)

        sheet.paste(tile, (left, top))
        caption_reason = row.get("caption_reason", row.get("source_id", ""))
        caption = f"{row['candidate_id'][:16]}|{caption_reason}|{row[score_name]:.3f}"
        draw.text((left + 3, top + TILE_SIZE + 4), caption, fill=CAPTION_COLOR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    sheet.save(temporary_path, format="JPEG", quality=SHEET_QUALITY)
    temporary_path.replace(output_path)

def write_watermark_sheets(score_path: Path, sheet_dir: Path) -> list[Path]:
    rows = pl.scan_parquet(score_path).sort("watermark_or_logo", descending=True).head(200).collect().to_dicts()
    paths = []

    for offset in range(0, len(rows), SHEET_SIZE):
        output_path = sheet_dir / f"watermark-{offset // SHEET_SIZE + 1:02d}.jpg"
        create_sheet(rows[offset:offset + SHEET_SIZE], output_path, "watermark_or_logo")
        paths.append(output_path)

    return paths

def write_audit_sheets(clean_dir: Path) -> tuple[int, int]:
    decision_path = clean_dir / "decisions.parquet"
    score_path = clean_dir / "score.parquet"

    if not decision_path.is_file() or not score_path.is_file():
        raise FileNotFoundError("audit sheets need decisions.parquet and score.parquet")

    frame = pl.read_parquet(decision_path).join(
        pl.read_parquet(score_path).select("candidate_id", "watermark_or_logo"),
        on="candidate_id",
        how="left"
    )
    frame = frame.with_columns(pl.col("watermark_or_logo").fill_null(0.0).alias("audit_score"))
    sheet_dir = clean_dir / "sheets"
    audit_rows = []
    sheet_count = 0

    for decision in ("drop", "keep"):
        subset = frame.filter(pl.col("decision") == decision)
        count = min(100, subset.height)
        rows = subset.sample(n=count, seed=42).sort("audit_score", descending=True).to_dicts()

        for offset in range(0, len(rows), SHEET_SIZE):
            sheet_count += 1
            sheet_rows = rows[offset:offset + SHEET_SIZE]

            for row in sheet_rows:
                row["caption_reason"] = decision
                audit_rows.append({"image_id": row["candidate_id"], "decision": "", "reason": "", "expected": decision})

            create_sheet(sheet_rows, sheet_dir / f"audit-{decision}-{offset // SHEET_SIZE + 1:02d}.jpg", "audit_score")

    pilot = frame.filter((pl.col("decision") == "review") & pl.col("source_id").is_in(PILOT_SOURCE_IDS))
    pilot = pilot.sample(n=min(400, pilot.height), seed=42).sort("audit_score", descending=True)
    pilot_rows = pilot.to_dicts()

    for offset in range(0, len(pilot_rows), SHEET_SIZE):
        sheet_count += 1
        sheet_rows = pilot_rows[offset:offset + SHEET_SIZE]

        for row in sheet_rows:
            row["caption_reason"] = "review"

        create_sheet(sheet_rows, sheet_dir / f"pilot-review-{offset // SHEET_SIZE + 1:02d}.jpg", "audit_score")

    audit_path = clean_dir / "audit.csv"
    temporary_path = audit_path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "decision", "reason", "expected"])
        writer.writeheader()
        writer.writerows(audit_rows)

    temporary_path.replace(audit_path)
    return sheet_count, len(audit_rows)

def summarize_audit(rows: list[dict]) -> str:
    dropped = []
    kept = []

    for row in rows:
        if row["decision"] not in ("keep", "drop"):
            raise ValueError(f"missing audit label for {row['image_id']}")

        if row["expected"] == "drop":
            dropped.append(row)
        elif row["expected"] == "keep":
            kept.append(row)

    correct_drops = sum(row["decision"] == "drop" for row in dropped)
    leaked_keeps = sum(row["decision"] == "drop" for row in kept)
    drop_lower, drop_upper = wilson_interval(correct_drops, len(dropped))
    leak_lower, leak_upper = wilson_interval(leaked_keeps, len(kept))
    lines = [
        "# Triage audit",
        "",
        f"Drop precision: {correct_drops}/{len(dropped)}; Wilson 95% {drop_lower:.3f}–{drop_upper:.3f}.",
        f"Keep leakage: {leaked_keeps}/{len(kept)}; Wilson 95% {leak_lower:.3f}–{leak_upper:.3f}.",
        "",
        "These estimates describe the sampled audit tiles only."
    ]
    return "\n".join(lines) + "\n"

def write_audit_report(clean_dir: Path) -> Path:
    audit_path = clean_dir / "audit.csv"

    if not audit_path.is_file():
        raise FileNotFoundError(f"missing audit labels: {audit_path}")

    with audit_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    report_path = clean_dir / "audit-report.md"
    temporary_path = report_path.with_suffix(".tmp")
    temporary_path.write_text(summarize_audit(rows), encoding="utf-8")
    temporary_path.replace(report_path)
    return report_path

def write_review_sheets(clean_dir: Path) -> tuple[int, int]:
    decision_path = clean_dir / "decisions.parquet"
    score_path = clean_dir / "score.parquet"

    if not decision_path.is_file() or not score_path.is_file():
        raise FileNotFoundError("review sheets need decisions.parquet and score.parquet")

    categories = [
        "watermark_or_logo", "text_poster_or_menu", "collage_or_grid", "screenshot_or_ui",
        "drawing_or_illustration", "not_food", "packaged_product", "person_only"
    ]
    scores = pl.scan_parquet(score_path).select("candidate_id", *categories)
    frame = pl.scan_parquet(decision_path).filter(pl.col("decision") == "review").join(
        scores,
        on="candidate_id",
        how="left"
    )
    frame = frame.with_columns(pl.max_horizontal(*categories).fill_null(0.0).alias("review_score"))
    rows = frame.sort("review_score", descending=True).collect().to_dicts()
    sheet_dir = clean_dir / "sheets"
    sheet_count = 0

    for offset in range(0, len(rows), SHEET_SIZE):
        sheet_count += 1
        sheet_rows = rows[offset:offset + SHEET_SIZE]

        for row in sheet_rows:
            row["caption_reason"] = row["source_id"]

        create_sheet(sheet_rows, sheet_dir / f"review-{sheet_count:04d}.jpg", "review_score")

    review_path = clean_dir / "review.csv"
    temporary_path = review_path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "decision", "reason"])
        writer.writeheader()

        for row in rows:
            writer.writerow({"image_id": row["candidate_id"], "decision": "", "reason": ""})

    temporary_path.replace(review_path)
    return sheet_count, len(rows)

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye data triage sheets")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    parser.add_argument("--kind", choices=["watermark", "audit", "audit-report", "review"], default="watermark")
    arguments = parser.parse_args(argv)
    clean_dir = arguments.data_root / "acquisition" / "clean"

    if arguments.kind == "audit":
        sheets, count = write_audit_sheets(clean_dir)
        return [f"wrote {sheets} audit sheets for {count} keep/drop tiles"]

    if arguments.kind == "audit-report":
        return [f"audit report {write_audit_report(clean_dir)}"]

    if arguments.kind == "review":
        sheets, count = write_review_sheets(clean_dir)
        return [f"wrote {sheets} review sheets for {count} images"]

    score_path = clean_dir / "score.parquet"

    if not score_path.is_file():
        raise FileNotFoundError(f"missing model scores: {score_path}")

    paths = write_watermark_sheets(score_path, clean_dir / "sheets")
    return [f"wrote {len(paths)} watermark sheets"]
