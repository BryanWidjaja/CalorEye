import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import polars as pl

from caloreye.paths import REPO_ROOT_DIR, get_data_root_dir
from caloreye.runs import start_run
from caloreye.taxonomy import CLASS_MAP_PATH, PILOT_SOURCE_IDS, read_class_map

PRECISION_TARGET = 0.95
MIN_CALIBRATION_COUNT = 10
CALIBRATION_SIZE = 300
KEEP_DUPLICATES = 3
DECISION_VERSION = "1"
DISABLED_CUTOFF = 1e9
TRUSTED_ROUTES = ("food101", "uec256")
CALIBRATION_SIGNALS = ("watermark_or_logo", "text_poster_or_menu", "not_food", "class_mismatch", "duplicates")
REVIEW_SIGNALS = ("watermark_or_logo", "text_poster_or_menu", "not_food", "artifact_margin", "mismatch_score")
REVIEW_FRACTION = 0.008
PILOT_REVIEW_FRACTION = 0.02
ARTIFACT_SIGNALS = ("collage_or_grid", "screenshot_or_ui", "drawing_or_illustration", "packaged_product", "person_only")

def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0

    estimate = successes / total
    variance = estimate * (1 - estimate) / total + 1.96 ** 2 / (4 * total ** 2)
    denominator = 1 + 1.96 ** 2 / total
    center = (estimate + 1.96 ** 2 / (2 * total)) / denominator
    radius = 1.96 * math.sqrt(variance) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)

def choose_cutoff(rows: list[dict]) -> tuple[float, int, int, int]:
    ordered = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    positives = 0
    best_cutoff = DISABLED_CUTOFF
    best_count = 0
    best_positives = 0

    for index, row in enumerate(ordered, start=1):
        if row["truth_decision"] == "drop":
            positives += 1

        if index < MIN_CALIBRATION_COUNT:
            continue

        if positives / index >= PRECISION_TARGET:
            best_cutoff = float(row["score"])
            best_count = index
            best_positives = positives

    return best_cutoff, best_count, best_positives, positives

def load_calibration(calibration_path: Path) -> list[dict]:
    if not calibration_path.is_file():
        raise FileNotFoundError(f"missing calibration labels: {calibration_path}")

    with calibration_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != CALIBRATION_SIZE:
        raise ValueError(f"expected 300 calibration labels, found {len(rows)}")

    for row in rows:
        if row["truth_decision"] not in ("keep", "drop"):
            raise ValueError(f"missing keep/drop calibration label for {row['candidate_id']}")

    return rows

def calibrate_thresholds(rows: list[dict], probe: pl.DataFrame) -> tuple[dict, list[str]]:
    thresholds = {}
    report = []

    for signal in CALIBRATION_SIGNALS:
        members = []

        for row in rows:
            if row["signal"] == signal:
                members.append(row)

        cutoff, count, positives, all_positives = choose_cutoff(members)
        thresholds[signal] = cutoff
        lower, upper = wilson_interval(positives, count)
        recall = positives / all_positives if all_positives else 0.0
        report.append(f"| {signal} | {cutoff:.4f} | {positives}/{count} | {lower:.3f}–{upper:.3f} | {recall:.3f} |")

    labelled = pl.DataFrame(rows).select("candidate_id", "truth_decision").join(
        probe.select("candidate_id", "text_density", "corner_overlay_score"),
        on="candidate_id"
    )
    bad = labelled.filter(pl.col("truth_decision") == "drop")
    thresholds["text_density"] = DISABLED_CUTOFF
    thresholds["corner_overlay_score"] = DISABLED_CUTOFF

    if bad.height:
        thresholds["text_density"] = float(bad["text_density"].quantile(0.25))
        thresholds["corner_overlay_score"] = float(bad["corner_overlay_score"].quantile(0.25))

    return thresholds, report

def build_review_cutoffs(frame: pl.DataFrame) -> dict:
    web = frame.filter(pl.col("discovery_route").str.starts_with("web"))
    pilot = web.filter(pl.col("source_id").is_in(PILOT_SOURCE_IDS))
    others = web.filter(~pl.col("source_id").is_in(PILOT_SOURCE_IDS))
    cutoffs = {"review_web": {}, "review_pilot": {}}

    for signal in REVIEW_SIGNALS:
        cutoffs["review_web"][signal] = float(others[signal].quantile(1 - REVIEW_FRACTION))
        cutoffs["review_pilot"][signal] = float(pilot[signal].quantile(1 - PILOT_REVIEW_FRACTION))

    return cutoffs

def review_reasons(row: dict, thresholds: dict) -> list[str]:
    cutoffs = thresholds["review_pilot"]
    reasons = []

    if row["source_id"] not in PILOT_SOURCE_IDS:
        cutoffs = thresholds["review_web"]

    for signal in REVIEW_SIGNALS:
        if row[signal] >= cutoffs[signal]:
            reasons.append(f"review_{signal}")

    return reasons

def decide_row(row: dict, thresholds: dict, multi_dish: set[str]) -> tuple[str, list[str]]:
    reasons = list(row["hard_reasons"] or [])

    if reasons:
        return "drop", reasons

    if row["model_revision"] is None:
        return "review", ["missing_model_score"]

    text_signal = row["text_density"] >= thresholds["text_density"]
    corner_signal = row["corner_overlay_score"] >= thresholds["corner_overlay_score"]
    watermark = row["watermark_or_logo"] >= thresholds["watermark_or_logo"]
    duplicate = row["cluster_rank"] > KEEP_DUPLICATES and row["embedding_cluster_size"] >= thresholds["duplicates"]

    if duplicate:
        reasons.append("near_duplicate_cap")

    if watermark and (text_signal or corner_signal):
        reasons.append("watermark")

    if reasons:
        return "drop", reasons

    is_trusted = row["discovery_route"] in TRUSTED_ROUTES

    if is_trusted:
        if watermark or row["cluster_rank"] > KEEP_DUPLICATES:
            return "review", ["trusted_suspect"]

        return "keep", []

    poster = row["text_poster_or_menu"] >= thresholds["text_poster_or_menu"]
    not_food = row["not_food"] >= thresholds["not_food"]
    wrong_class = -row["class_margin"] >= thresholds["class_mismatch"]
    is_multi_dish = row["source_id"] in multi_dish
    is_label_suspect = row["confident_learning_suspect"]

    if poster and text_signal:
        reasons.append("text_poster_or_menu")

    if not_food and is_label_suspect and row["class_margin"] < 0:
        reasons.append("not_food")

    if wrong_class and is_label_suspect and not is_multi_dish:
        reasons.append("wrong_class")

    if reasons:
        return "drop", reasons

    reasons = review_reasons(row, thresholds)

    if reasons:
        return "review", reasons

    return "keep", []

def build_report(decisions: pl.DataFrame, calibration_report: list[str]) -> str:
    counts = Counter(decisions["decision"].to_list())
    lines = ["# Acquisition triage report", "", f"Total: {decisions.height}; keep: {counts['keep']}; drop: {counts['drop']}; review: {counts['review']}", "", "## Calibration", "", "Signal | Cutoff | Precision | Wilson 95% | Sample recall", "--- | ---: | ---: | ---: | ---:"]
    lines.extend(calibration_report)
    lines.extend(["", "Recall is measured only within the deliberately selected calibration sheets.", "", "## Pilot folders", "", "Folder | Keep | Drop | Review", "--- | ---: | ---: | ---:"])

    for source_id in PILOT_SOURCE_IDS:
        subset = decisions.filter(pl.col("source_id") == source_id)
        pilot_counts = Counter(subset["decision"].to_list())
        lines.append(f"{source_id} | {pilot_counts['keep']} | {pilot_counts['drop']} | {pilot_counts['review']}")

    lines.extend(["", "## Counts by folder and decision", "", "Folder | Decision | Count", "--- | --- | ---:"])

    for source_id, decision, count in decisions.group_by("source_id", "decision").len().sort("source_id", "decision").iter_rows():
        lines.append(f"{source_id} | {decision} | {count}")

    lines.extend(["", "## Counts by reason", "", "Reason | Count", "--- | ---:"])
    reasons = Counter()

    for row_reasons in decisions["reasons"]:
        reasons.update(row_reasons)

    for reason, count in sorted(reasons.items()):
        lines.append(f"{reason} | {count}")

    return "\n".join(lines) + "\n"

def run_decision(data_root: Path) -> tuple[Path, Counter]:
    clean_dir = data_root / "acquisition" / "clean"
    calibration_path = REPO_ROOT_DIR / "notes" / "research" / "data-clean" / "calibration.csv"
    calibration = load_calibration(calibration_path)
    probe = pl.read_parquet(clean_dir / "probe.parquet")
    thresholds, calibration_report = calibrate_thresholds(calibration, probe)
    start_run("data-triage-decide", {"version": DECISION_VERSION, "thresholds": thresholds})
    score = pl.read_parquet(clean_dir / "score.parquet").select(
        "candidate_id", "model_revision", "watermark_or_logo", "text_poster_or_menu", "not_food",
        "class_margin", "given_class_score", *ARTIFACT_SIGNALS
    )
    agree = pl.read_parquet(clean_dir / "agree.parquet").select(
        "candidate_id", "confident_learning_suspect", "embedding_cluster", "embedding_cluster_size"
    )
    frame = probe.join(score, on="candidate_id", how="left").join(agree, on="candidate_id", how="left")
    frame = frame.sort("embedding_cluster", "width", "height", descending=[False, True, True], nulls_last=True)
    frame = frame.with_columns(pl.col("candidate_id").cum_count().over("embedding_cluster").alias("cluster_rank"))
    multi_dish = set()

    for class_row in read_class_map(CLASS_MAP_PATH):
        if class_row.plan == "comp":
            multi_dish.add(class_row.source_id)

    rows = []

    for row in frame.iter_rows(named=True):
        decision, reasons = decide_row(row, thresholds, multi_dish)
        rows.append({
            "candidate_id": row["candidate_id"],
            "source_id": row["source_id"],
            "discovery_route": row["discovery_route"],
            "local_path": row["local_path"],
            "original_sha256": row["original_sha256"],
            "decision": decision,
            "reasons": reasons,
            "version": DECISION_VERSION
        })

    decisions = pl.DataFrame(rows).sort("candidate_id")
    output_path = clean_dir / "decisions.parquet"
    temporary_path = output_path.with_suffix(".tmp")
    decisions.write_parquet(temporary_path)
    temporary_path.replace(output_path)
    threshold_path = clean_dir / "thresholds.json"
    temporary_path = threshold_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(threshold_path)
    report_path = clean_dir / "report.md"
    temporary_path = report_path.with_suffix(".tmp")
    temporary_path.write_text(build_report(decisions, calibration_report), encoding="utf-8")
    temporary_path.replace(report_path)
    return output_path, Counter(decisions["decision"].to_list())

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye data triage decide")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    arguments = parser.parse_args(argv)
    output_path, counts = run_decision(arguments.data_root)
    return [f"keep {counts['keep']} drop {counts['drop']} review {counts['review']}", f"decisions {output_path}"]
