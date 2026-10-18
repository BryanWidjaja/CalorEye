import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from caloreye.data.clean import find_latest_flags
from caloreye.data.manifest import DATE_FORMAT, LATEST_MANIFEST_NAME, WORK_DIR_NAME
from caloreye.paths import get_data_root_dir
from caloreye.runs import start_run
from caloreye.taxonomy import PILOT_SOURCE_IDS

VALIDATION_FRACTION = 0.15
HOST_LINK_MAX_IMAGES = 100
SPLIT_SEED = 0
TRAIN_NAME = "train"
VAL_NAME = "val"
TEST_OWN_NAME = "test_own"
FROZEN_NAME = "splits-frozen.json"
SPLITS_PREFIX = "splits"
JSON_INDENT = 2
LARGE_GROUP_FRACTION = 0.05

def find_root(parents: list[int], node: int) -> int:
    root = node

    while parents[root] != root:
        root = parents[root]

    while parents[node] != root:
        next_node = parents[node]
        parents[node] = root
        node = next_node

    return root

def build_groups(clusters: list[int], hosts: list[str], host_max: int) -> list[int]:
    parents = list(range(len(clusters)))
    host_counts = Counter(hosts)
    first_row = {}

    for row, cluster in enumerate(clusters):
        link_keys = [("c", cluster)]

        if hosts[row] and host_counts[hosts[row]] <= host_max:
            link_keys.append(("h", hosts[row]))

        for key in link_keys:
            if key not in first_row:
                first_row[key] = row
                continue

            root_a = find_root(parents, row)
            root_b = find_root(parents, first_row[key])

            if root_a != root_b:
                parents[max(root_a, root_b)] = min(root_a, root_b)

    return [find_root(parents, row) for row in range(len(clusters))]

def count_by_group(groups: list[int], folders: list[str]) -> dict[int, Counter]:
    per_group = {}

    for group, folder in zip(groups, folders, strict=True):
        per_group.setdefault(group, Counter())[folder] += 1

    return per_group

def assign_groups(
        per_group: dict[int, Counter],
        totals: Counter,
        fraction: float,
        seed: int
    ) -> dict[int, str]:
    needs = {folder: round(fraction * total) for folder, total in totals.items()}
    order = sorted(per_group, key=lambda group: -sum(per_group[group].values()))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda group: -sum(per_group[group].values()))
    assignment = {}

    for group in order:
        gain = 0

        for folder, count in per_group[group].items():
            if needs[folder] >= count:
                gain += count
            else:
                gain += needs[folder] - (count - needs[folder])

        if gain > 0:
            assignment[group] = VAL_NAME

            for folder, count in per_group[group].items():
                needs[folder] -= count
        else:
            assignment[group] = TRAIN_NAME

    return assignment

def find_leaks(frame: pl.DataFrame, host_max: int) -> list[str]:
    problems = []
    cluster_spans = frame.group_by("dup_cluster").agg(pl.col("split").n_unique().alias("spans"))
    leaking_clusters = cluster_spans.filter(pl.col("spans") > 1).height

    if leaking_clusters:
        problems.append(f"{leaking_clusters} duplicate clusters span both splits")

    hosts = frame.filter(pl.col("host") != "").group_by("host").agg(
        pl.len().alias("images"),
        pl.col("split").n_unique().alias("spans")
    )
    linked = hosts.filter((pl.col("images") <= host_max) & (pl.col("images") > 1))
    leaking_hosts = linked.filter(pl.col("spans") > 1).height

    if leaking_hosts:
        problems.append(f"{leaking_hosts} linked hosts span both splits")

    return problems

def compute_split_hash(frame: pl.DataFrame) -> str:
    pairs = sorted(zip(frame["image_id"].to_list(), frame["split"].to_list(), strict=True))
    digest = hashlib.sha256()

    for image_id, split_name in pairs:
        digest.update(f"{image_id}:{split_name}\n".encode())

    return digest.hexdigest()

def load_kept_rows(work_dir: Path) -> pl.DataFrame:
    flags = pl.read_parquet(find_latest_flags(work_dir)).filter(pl.col("keep"))
    manifest = pl.read_parquet(work_dir / LATEST_MANIFEST_NAME)

    if "triage_decision" in manifest.columns:
        manifest = manifest.filter(pl.col("triage_decision") == "keep")
        flags = flags.join(manifest.select("path"), on="path", how="inner")

    columns = manifest.select(["path", "host", "dup_cluster"])

    return flags.select(["image_id", "path", "source_id"]).join(columns, on="path", how="left")

def build_split_frame(rows: pl.DataFrame, host_max: int, seed: int) -> pl.DataFrame:
    clusters = rows["dup_cluster"].to_list()
    hosts = rows["host"].fill_null("").to_list()
    folders = rows["source_id"].to_list()
    groups = build_groups(clusters, hosts, host_max)
    per_group = count_by_group(groups, folders)
    assignment = assign_groups(per_group, Counter(folders), VALIDATION_FRACTION, seed)
    splits = [assignment[group] for group in groups]

    return rows.with_columns(
        pl.Series("group_id", groups),
        pl.Series("split", splits)
    )

def report_lines(frame: pl.DataFrame) -> list[str]:
    lines = ["pilot class split (train / val)"]

    for source_id in PILOT_SOURCE_IDS:
        folder = frame.filter(pl.col("source_id") == source_id)
        train_count = folder.filter(pl.col("split") == TRAIN_NAME).height
        val_count = folder.filter(pl.col("split") == VAL_NAME).height
        lines.append(f"{source_id}\t{train_count}\t{val_count}")

    totals = frame.group_by("split").len().sort("split")

    for split_name, count in totals.iter_rows():
        lines.append(f"all folders {split_name} {count}")

    group_sizes = frame.group_by("group_id").len().sort("len", descending=True)
    large = group_sizes.filter(pl.col("len") > LARGE_GROUP_FRACTION * frame.height)
    lines.append(f"groups {group_sizes.height}  largest {group_sizes['len'].max()}")
    lines.append(f"groups above {LARGE_GROUP_FRACTION:.0%} of images: {large.height}")
    lines.append(f"{TEST_OWN_NAME} reserved, 0 web images")

    return lines

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye data split")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    parser.add_argument("--host-link-max", type=int, default=HOST_LINK_MAX_IMAGES)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args(argv)
    work_dir = arguments.data_root / WORK_DIR_NAME
    frozen_path = work_dir / FROZEN_NAME

    if frozen_path.is_file() and not arguments.force:
        raise FileExistsError(f"expected no frozen split at {frozen_path}, found one; pass --force")

    rows = load_kept_rows(work_dir)
    frame = build_split_frame(rows, arguments.host_link_max, arguments.seed)
    problems = find_leaks(frame, arguments.host_link_max)

    if problems:
        raise ValueError(f"expected no leakage, found {problems}")

    stamp = datetime.now(UTC).strftime(DATE_FORMAT)
    splits_path = work_dir / f"{SPLITS_PREFIX}-{stamp}.parquet"
    frame.write_parquet(splits_path)
    split_hash = compute_split_hash(frame)
    frozen = {"split_hash": split_hash, "file": splits_path.name, "rows": frame.height}
    frozen_path.write_text(json.dumps(frozen, indent=JSON_INDENT) + "\n", encoding="utf-8")
    start_run(
        "data-split",
        {
            "seed": arguments.seed,
            "host_link_max": arguments.host_link_max,
            "split_hash": split_hash
        }
    )

    return report_lines(frame) + [f"split sha256 {split_hash}", f"wrote {splits_path}"]
