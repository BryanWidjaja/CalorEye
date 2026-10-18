import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from caloreye.paths import get_data_root_dir
from caloreye.runs import set_seed, start_run
from caloreye.stage2 import ood_scores
from caloreye.stage2.dataset import (
    SPLIT_TRAIN,
    SPLIT_VAL,
    CropDataset,
    build_cache,
    build_unseen_box_rows,
    build_unseen_rows,
    load_split_rows,
    sampler_weights,
    split_unseen_folders,
)
from caloreye.stage2.train import (
    BEST_NAME,
    MODEL_DIR_NAME,
    SEED,
    collect_logits,
    create_model,
    fit_model,
    make_loader,
)

OTHER_CLASS_INDEX = 7
OTHER_CLASS_COUNT = 8
PILOT_CLASS_COUNT = 6
OTHER_TRAIN_CAP = 2500
OTHER_VAL_SHARE = 0.1
OTHER_SEED = 0
REPORT_NAME = "efficientnet-b0-other-class.report.json"
OOD_REPORT_NAME = "efficientnet-b0-seven-trained.ood.json"
BASELINE_NAME = "max_softmax"
JSON_INDENT = 2

def select_rows_by_folders(rows: list[dict], folders: set[str]) -> list[dict]:
    selected = []

    for row in rows:
        if row["source_id"] in folders:
            selected.append(row)

    return selected

def relabel_as_other(rows: list[dict], split_name: str) -> list[dict]:
    relabelled = []

    for row in rows:
        other_row = dict(row)
        other_row["dish_idx"] = OTHER_CLASS_INDEX
        other_row["split"] = split_name
        relabelled.append(other_row)

    return relabelled

def split_other_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    image_ids = sorted({row["image_id"] for row in rows})
    random.Random(OTHER_SEED).shuffle(image_ids)
    val_ids = set(image_ids[: int(OTHER_VAL_SHARE * OTHER_TRAIN_CAP)])
    shuffled_rows = list(rows)
    random.Random(OTHER_SEED).shuffle(shuffled_rows)
    train_rows = []
    val_rows = []

    for row in shuffled_rows:
        if row["image_id"] in val_ids:
            val_rows.append(row)
        elif len(train_rows) < OTHER_TRAIN_CAP:
            train_rows.append(row)

    return train_rows, val_rows

def collect_unseen_views(data_root: Path) -> dict:
    return {
        "unseen_box": build_unseen_box_rows(data_root),
        "unseen_whole": build_unseen_rows(data_root)
    }

def compute_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))

    return shifted / shifted.sum(axis=1, keepdims=True)

def evaluate_other_model(
        model: torch.nn.Module,
        data_root: Path,
        val_rows: list[dict],
        test_views: dict
    ) -> dict:
    val_logits, val_labels = collect_logits(
        model,
        make_loader(CropDataset(data_root, val_rows, False, SEED), False, None)
    )
    val_probabilities = compute_softmax(val_logits)
    is_known = val_labels < PILOT_CLASS_COUNT
    known_scores = val_probabilities[is_known][:, :PILOT_CLASS_COUNT].max(axis=1)
    known_predictions = val_probabilities[is_known].argmax(axis=1)
    report = {
        "known_count": int(is_known.sum()),
        "known_accuracy": float((known_predictions == val_labels[is_known]).mean()),
        "known_predicted_other": float((known_predictions == OTHER_CLASS_INDEX).mean())
    }

    for view_name, rows in test_views.items():
        logits, _ = collect_logits(
            model,
            make_loader(CropDataset(data_root, rows, False, SEED), False, None)
        )
        probabilities = compute_softmax(logits)
        unseen_scores = probabilities[:, :PILOT_CLASS_COUNT].max(axis=1)
        summary = ood_scores.summarize_scores(known_scores, unseen_scores)
        summary["count"] = len(rows)
        predicted_other = probabilities.argmax(axis=1) == OTHER_CLASS_INDEX
        summary["predicted_other"] = float(predicted_other.mean())
        report[view_name] = summary

    return report

def read_baseline(data_root: Path) -> dict:
    report_path = data_root / MODEL_DIR_NAME / OOD_REPORT_NAME
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    baseline = {}

    for view_name in ("unseen_box", "unseen_whole"):
        baseline[view_name] = payload["methods"][view_name][BASELINE_NAME]["test"]

    return baseline

def run_other_experiment(data_root: Path) -> list[str]:
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    run_dir = start_run("stage2-other-class", {"seed": SEED, "cap": OTHER_TRAIN_CAP}).run_dir
    views = collect_unseen_views(data_root)
    unseen_ids = [row["source_id"] for row in views["unseen_whole"]]
    unseen_ids.extend(row["source_id"] for row in views["unseen_box"])
    fit_folders, test_folders = split_unseen_folders(unseen_ids)
    fit_rows = select_rows_by_folders(views["unseen_whole"] + views["unseen_box"], fit_folders)
    other_train, other_val = split_other_rows(relabel_as_other(fit_rows, SPLIT_TRAIN))
    train_rows = load_split_rows(data_root, SPLIT_TRAIN) + other_train
    val_rows = load_split_rows(data_root, SPLIT_VAL) + other_val
    test_views = {}

    for view_name, rows in views.items():
        test_views[view_name] = select_rows_by_folders(rows, test_folders)

    build_cache(data_root, train_rows + val_rows)
    train_loader = make_loader(
        CropDataset(data_root, train_rows, True, SEED),
        True,
        sampler_weights(train_rows, OTHER_CLASS_COUNT)
    )
    val_loader = make_loader(CropDataset(data_root, val_rows, False, SEED), False, None)
    model = create_model(OTHER_CLASS_COUNT)
    history = fit_model(model, train_loader, val_loader, run_dir, OTHER_CLASS_COUNT)
    model.load_state_dict(torch.load(run_dir / BEST_NAME))
    report = {
        "data_slice": "web_val",
        "other_train_rows": len(other_train),
        "other_val_rows": len(other_val),
        "fit_folder_count": len(fit_folders),
        "test_folder_count": len(test_folders),
        "history": history,
        "other_class_model": evaluate_other_model(model, data_root, val_rows, test_views),
        "max_softmax_baseline": read_baseline(data_root)
    }
    output_path = data_root / MODEL_DIR_NAME / REPORT_NAME
    output_path.write_text(json.dumps(report, indent=JSON_INDENT), encoding="utf-8")

    return [f"other class model {report['other_class_model']}", f"wrote {output_path}"]

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye stage2 other")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    arguments = parser.parse_args(argv)

    return run_other_experiment(arguments.data_root)
