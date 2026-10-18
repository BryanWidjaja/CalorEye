import argparse
import json
import shutil
from pathlib import Path

from caloreye.paths import get_data_root_dir
from caloreye.runs import start_run

WEB_DIR_NAME = "acquisition_web"
ACQUISITION_DIR_NAME = "acquisition"
CANDIDATES_NAME = "candidates.jsonl"
BACKUP_NAME = "candidates.pre-merge.jsonl"
IMAGES_DIR_NAME = "images"

def read_records(candidates_path: Path) -> list[dict]:
    records = []

    with candidates_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    return records

def find_source_file(record: dict, web_dir: Path) -> Path:
    file_name = Path(record["local_path"].replace("\\", "/")).name

    return web_dir / IMAGES_DIR_NAME / record["source_id"] / file_name

def merge_acquisition(web_dir: Path, acquisition_dir: Path) -> dict:
    candidates_path = acquisition_dir / CANDIDATES_NAME
    backup_path = acquisition_dir / BACKUP_NAME

    if not backup_path.exists():
        shutil.copyfile(candidates_path, backup_path)

    existing = read_records(candidates_path)
    known_hashes = {record["original_sha256"] for record in existing}
    known_urls = {record["original_image_url"] for record in existing}
    merged = []
    counts = {"web_rows": 0, "skipped_duplicate": 0, "skipped_missing_file": 0, "merged": 0, "bytes": 0}
    per_source = {}

    for record in read_records(web_dir / CANDIDATES_NAME):
        counts["web_rows"] += 1

        if record["original_sha256"] in known_hashes or record["original_image_url"] in known_urls:
            counts["skipped_duplicate"] += 1
            continue

        source_file = find_source_file(record, web_dir)

        if not source_file.is_file():
            counts["skipped_missing_file"] += 1
            continue

        target_file = acquisition_dir / IMAGES_DIR_NAME / record["source_id"] / source_file.name
        target_file.parent.mkdir(parents=True, exist_ok=True)

        if not target_file.is_file():
            temporary_file = target_file.with_suffix(".tmp")
            shutil.copyfile(source_file, temporary_file)
            temporary_file.replace(target_file)

        record["local_path"] = str(target_file)
        merged.append(record)
        known_hashes.add(record["original_sha256"])
        known_urls.add(record["original_image_url"])
        counts["merged"] += 1
        counts["bytes"] += target_file.stat().st_size
        per_source[record["source_id"]] = per_source.get(record["source_id"], 0) + 1

    temporary_path = candidates_path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in existing + merged:
            handle.write(json.dumps(record) + "\n")

    temporary_path.replace(candidates_path)
    counts["per_source"] = per_source

    return counts

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye data merge")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    arguments = parser.parse_args(argv)
    start_run("data-merge-acquisition", {"data_root": str(arguments.data_root)})
    counts = merge_acquisition(arguments.data_root / WEB_DIR_NAME, arguments.data_root / ACQUISITION_DIR_NAME)
    pilot = {key: value for key, value in counts["per_source"].items() if key in ("6", "5", "39", "B1", "11", "1")}

    return [
        f"web rows {counts['web_rows']}, merged {counts['merged']}, duplicates {counts['skipped_duplicate']}, missing {counts['skipped_missing_file']}",
        f"merged bytes {counts['bytes']}",
        f"pilot merged {pilot}"
    ]
