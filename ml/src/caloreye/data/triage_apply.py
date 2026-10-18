import argparse
import csv
import json
import shutil
from pathlib import Path

import polars as pl

from caloreye.paths import get_data_root_dir
from caloreye.runs import hash_file, start_run

REVIEW_DECISIONS = ("keep", "drop")

def read_reviews(review_path: Path, decisions: pl.DataFrame) -> dict[str, tuple[str, str]]:
    pending = set(decisions.filter(pl.col("decision") == "review")["candidate_id"])
    updates = {}

    with review_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            candidate_id = row["image_id"]
            decision = row["decision"].strip().lower()

            if candidate_id not in pending or decision not in REVIEW_DECISIONS:
                raise ValueError(f"invalid review decision for {candidate_id}")

            if candidate_id in updates:
                raise ValueError(f"duplicate review decision for {candidate_id}")

            updates[candidate_id] = decision, row["reason"].strip()

    return updates

def apply_reviews(decision_path: Path, review_path: Path) -> pl.DataFrame:
    decisions = pl.read_parquet(decision_path)
    updates = read_reviews(review_path, decisions)
    revised = []

    for row in decisions.iter_rows(named=True):
        if row["candidate_id"] in updates:
            decision, reason = updates[row["candidate_id"]]
            row["decision"] = decision
            row["reasons"] = [reason or "manual_review"]

        revised.append(row)

    result = pl.DataFrame(revised, schema=decisions.schema)
    return result

def join_manifest(manifest: pl.DataFrame, decisions: pl.DataFrame, acquisition_dir: Path) -> pl.DataFrame:
    paths = []

    for local_path in decisions["local_path"]:
        paths.append(Path(local_path).relative_to(acquisition_dir).as_posix())

    flags = decisions.select(
        pl.Series("path", paths),
        pl.col("decision").alias("triage_decision"),
        pl.col("reasons").alias("triage_reasons")
    )
    prior = manifest.drop("triage_decision", "triage_reasons", strict=False)
    joined = prior.join(flags, on="path", how="left")

    if joined.height != manifest.height or joined["triage_decision"].null_count():
        raise ValueError("manifest and decisions differ; rebuild manifest or rerun triage")

    return joined

def write_keep_list(decisions: pl.DataFrame, output_path: Path) -> None:
    temporary_path = output_path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in decisions.filter(pl.col("decision") == "keep").iter_rows(named=True):
            item = {"candidate_id": row["candidate_id"], "source_id": row["source_id"], "local_path": row["local_path"]}
            handle.write(json.dumps(item) + "\n")

    temporary_path.replace(output_path)

def apply_decisions(data_root: Path, review_path: Path | None) -> tuple[int, int]:
    acquisition_dir = data_root / "acquisition"
    clean_dir = acquisition_dir / "clean"
    decision_path = clean_dir / "decisions.parquet"
    manifest_path = data_root / "work" / "manifest.parquet"
    backup_path = clean_dir / "manifest.before-triage.parquet"
    marker_path = clean_dir / "applied.json"

    if not decision_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("apply needs decisions.parquet and work/manifest.parquet")

    decisions = pl.read_parquet(decision_path)

    if review_path is not None:
        decisions = apply_reviews(decision_path, review_path)

    manifest = pl.read_parquet(manifest_path)

    if manifest.height != decisions.height:
        raise ValueError(f"manifest has {manifest.height} rows but decisions have {decisions.height}; rebuild manifest")

    joined = join_manifest(manifest, decisions, acquisition_dir)

    if review_path is not None:
        temporary_path = decision_path.with_suffix(".tmp")
        decisions.write_parquet(temporary_path)
        temporary_path.replace(decision_path)

    if not marker_path.exists() and not backup_path.exists():
        temporary_path = backup_path.with_suffix(".tmp")
        shutil.copyfile(manifest_path, temporary_path)
        temporary_path.replace(backup_path)

    if not backup_path.is_file():
        raise FileNotFoundError(f"missing undo backup: {backup_path}")

    write_keep_list(decisions, clean_dir / "keep_list.jsonl")
    temporary_path = manifest_path.with_suffix(".tmp")
    joined.write_parquet(temporary_path)
    temporary_path.replace(manifest_path)
    marker = {"decisions_sha256": hash_file(decision_path), "manifest_backup": str(backup_path)}
    temporary_path = marker_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(marker_path)
    start_run("data-triage-apply", marker)
    return decisions.filter(pl.col("decision") == "keep").height, decisions.filter(pl.col("decision") == "drop").height

def undo_decisions(data_root: Path) -> Path:
    clean_dir = data_root / "acquisition" / "clean"
    backup_path = clean_dir / "manifest.before-triage.parquet"

    if not backup_path.is_file():
        raise FileNotFoundError(f"missing undo backup: {backup_path}")

    manifest_path = data_root / "work" / "manifest.parquet"
    temporary_path = manifest_path.with_suffix(".tmp")
    shutil.copyfile(backup_path, temporary_path)
    temporary_path.replace(manifest_path)
    (clean_dir / "applied.json").unlink(missing_ok=True)
    (clean_dir / "keep_list.jsonl").unlink(missing_ok=True)
    start_run("data-triage-undo", {"restored": str(backup_path)})
    backup_path.unlink()
    return manifest_path

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye data triage apply|undo")
    parser.add_argument("step", choices=["apply", "undo"])
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    parser.add_argument("--review", type=Path)
    arguments = parser.parse_args(argv)

    if arguments.step == "undo":
        return [f"restored {undo_decisions(arguments.data_root)}"]

    kept, dropped = apply_decisions(arguments.data_root, arguments.review)
    return [f"applied triage: {kept} keep, {dropped} drop"]
