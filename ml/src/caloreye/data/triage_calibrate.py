import argparse
import csv
from pathlib import Path

import polars as pl

from caloreye.data.triage_sheets import create_sheet
from caloreye.paths import REPO_ROOT_DIR, get_data_root_dir
from caloreye.taxonomy import PILOT_SOURCE_IDS

SAMPLE_PER_SIGNAL = 50
SIGNALS = ("watermark_or_logo", "text_poster_or_menu", "not_food", "class_mismatch", "duplicates", "pilot")
CALIBRATION_COLUMNS = ("candidate_id", "source_id", "signal", "score", "truth_decision", "truth_reason")

def sample_signal(frame: pl.LazyFrame, signal: str, seen: set[str]) -> list[dict]:
    if signal == "class_mismatch":
        candidates = frame.filter(pl.col("discovery_route").str.starts_with("web"))
        candidates = candidates.with_columns((-pl.col("class_margin")).alias("calibration_score"))
    elif signal == "duplicates":
        candidates = frame.with_columns(pl.col("embedding_cluster_size").cast(pl.Float64).alias("calibration_score"))
    elif signal == "pilot":
        candidates = frame.filter(pl.col("source_id").is_in(PILOT_SOURCE_IDS))
        candidates = candidates.with_columns(pl.col("candidate_id").hash(seed=42).cast(pl.Float64).alias("calibration_score"))
    else:
        candidates = frame.filter(pl.col("discovery_route").str.starts_with("web"))
        candidates = candidates.with_columns(pl.col(signal).alias("calibration_score"))

    rows = candidates.sort("calibration_score", descending=True).head(500).collect().to_dicts()
    selected = []

    for row in rows:
        if row["candidate_id"] in seen:
            continue

        selected.append(row)
        seen.add(row["candidate_id"])

        if len(selected) == SAMPLE_PER_SIGNAL:
            break

    if len(selected) != SAMPLE_PER_SIGNAL:
        raise ValueError(f"only {len(selected)} unique calibration images for {signal}")

    return selected

def write_calibration(data_root: Path) -> tuple[Path, int]:
    clean_dir = data_root / "acquisition" / "clean"
    score_path = clean_dir / "score.parquet"
    agree_path = clean_dir / "agree.parquet"

    if not score_path.is_file() or not agree_path.is_file():
        raise FileNotFoundError("calibration needs score.parquet and agree.parquet")

    csv_path = REPO_ROOT_DIR / "notes" / "research" / "data-clean" / "calibration.csv"

    if csv_path.exists():
        raise FileExistsError(f"calibration already exists: {csv_path}")

    frame = pl.scan_parquet(score_path).join(
        pl.scan_parquet(agree_path).select("candidate_id", "embedding_cluster_size"),
        on="candidate_id"
    )
    seen = set()
    all_rows = []
    sheet_dir = clean_dir / "sheets"

    for index, signal in enumerate(SIGNALS, start=1):
        rows = sample_signal(frame, signal, seen)
        create_sheet(rows, sheet_dir / f"calibration-{index:02d}.jpg", "calibration_score")

        for row in rows:
            all_rows.append({
                "candidate_id": row["candidate_id"],
                "source_id": row["source_id"],
                "signal": signal,
                "score": row["calibration_score"],
                "truth_decision": "",
                "truth_reason": ""
            })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = csv_path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CALIBRATION_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    temporary_path.replace(csv_path)
    return csv_path, len(all_rows)

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye data triage calibration-sheets")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    arguments = parser.parse_args(argv)
    output_path, count = write_calibration(arguments.data_root)
    return [f"wrote {count} calibration tiles", f"calibration {output_path}"]
