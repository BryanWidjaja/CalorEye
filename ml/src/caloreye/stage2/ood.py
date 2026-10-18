import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from caloreye.data.export_s2 import NOT_FOOD_INDEX
from caloreye.data.manifest import WORK_DIR_NAME
from caloreye.paths import get_data_root_dir
from caloreye.runs import start_run
from caloreye.stage2 import ood_scores
from caloreye.stage2.calibrate import CALIBRATION_NAME
from caloreye.stage2.dataset import (
    LOOKALIKE_SOURCE_IDS,
    SPLIT_TRAIN,
    SPLIT_VAL,
    CropDataset,
    build_cache,
    build_unseen_box_rows,
    build_unseen_rows,
    load_split_rows,
    split_unseen_folders,
)
from caloreye.stage2.export import load_trained_classifier
from caloreye.stage2.train import MODEL_DIR_NAME, TRAINED_WEIGHTS_NAME

FEATURE_CACHE_NAME = "ood_features.npz"
REPORT_NAME = "efficientnet-b0-seven-trained.ood.json"
VIM_DIMENSIONS = (64, 128, 256, 512)
BATCH_SIZE = 128
WORKER_COUNT = 2
BASELINE_NAME = "max_softmax"
SELECTION_COVERAGES = (0.95, 0.9)
TRIGGER_COVERAGE = 0.9
TRIGGER_MARGIN = 0.05
JSON_INDENT = 2
GROUP_NAMES = ("train", "val", "unseen_whole", "unseen_box")

def extract_group(model: torch.nn.Module, data_root: Path, rows: list[dict]) -> dict:
    loader = DataLoader(
        CropDataset(data_root, rows, False, 0),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKER_COUNT,
        pin_memory=True
    )
    feature_parts = []
    logit_parts = []
    label_parts = []

    with torch.inference_mode():
        for images, labels, _ in loader:
            pooled = model.forward_head(
                model.forward_features(images.cuda(non_blocking=True)),
                pre_logits=True
            )
            feature_parts.append(pooled.cpu().numpy())
            logit_parts.append(model.classifier(pooled).cpu().numpy())
            label_parts.append(labels.numpy())

    return {
        "features": np.concatenate(feature_parts),
        "logits": np.concatenate(logit_parts),
        "labels": np.concatenate(label_parts),
        "source_ids": np.array([row["source_id"] for row in rows])
    }

def select_dish_rows(rows: list[dict]) -> list[dict]:
    dish_rows = []

    for row in rows:
        if 0 <= row["dish_idx"] < NOT_FOOD_INDEX:
            dish_rows.append(row)

    return dish_rows

def collect_group_rows(data_root: Path) -> dict:
    train_rows = select_dish_rows(load_split_rows(data_root, SPLIT_TRAIN))
    val_rows = select_dish_rows(load_split_rows(data_root, SPLIT_VAL))
    unseen_whole_rows = build_unseen_rows(data_root)
    unseen_box_rows = build_unseen_box_rows(data_root)
    build_cache(data_root, train_rows + val_rows + unseen_whole_rows + unseen_box_rows)

    return {
        "train": train_rows,
        "val": val_rows,
        "unseen_whole": unseen_whole_rows,
        "unseen_box": unseen_box_rows
    }

def load_or_extract_groups(data_root: Path) -> dict:
    cache_path = data_root / WORK_DIR_NAME / FEATURE_CACHE_NAME

    if cache_path.is_file():
        stored = np.load(cache_path)
        groups = {}

        for group_name in GROUP_NAMES:
            groups[group_name] = {
                "features": stored[f"{group_name}_features"],
                "logits": stored[f"{group_name}_logits"],
                "labels": stored[f"{group_name}_labels"],
                "source_ids": stored[f"{group_name}_source_ids"]
            }

        return groups

    model = load_trained_classifier(data_root / MODEL_DIR_NAME / TRAINED_WEIGHTS_NAME).cuda()
    group_rows = collect_group_rows(data_root)
    groups = {}
    arrays = {}

    for group_name in GROUP_NAMES:
        groups[group_name] = extract_group(model, data_root, group_rows[group_name])

        for field_name, values in groups[group_name].items():
            arrays[f"{group_name}_{field_name}"] = values

    np.savez_compressed(cache_path, **arrays)

    return groups

def make_folder_mask(source_ids: np.ndarray, folders: set[str]) -> np.ndarray:
    return np.isin(source_ids, list(folders))

def read_temperature(data_root: Path) -> float:
    calibration_path = data_root / MODEL_DIR_NAME / CALIBRATION_NAME
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))

    return float(payload["temperature"])

def read_classifier_head(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    model = load_trained_classifier(data_root / MODEL_DIR_NAME / TRAINED_WEIGHTS_NAME)
    weights = model.classifier.weight.detach().numpy()
    bias = model.classifier.bias.detach().numpy()

    return weights, bias

def make_max_softmax_scorer(temperature: float):
    def score(features: np.ndarray, logits: np.ndarray) -> np.ndarray:
        return ood_scores.compute_max_softmax(logits, temperature)

    return score

def score_max_logit(features: np.ndarray, logits: np.ndarray) -> np.ndarray:
    return ood_scores.compute_max_logit(logits)

def score_energy(features: np.ndarray, logits: np.ndarray) -> np.ndarray:
    return ood_scores.compute_log_sum_exp(logits)

def make_vim_scorer(vim_model: ood_scores.VimModel):
    def score(features: np.ndarray, logits: np.ndarray) -> np.ndarray:
        return ood_scores.score_vim(vim_model, features, logits)

    return score

def build_score_functions(
        groups: dict,
        temperature: float,
        head: tuple[np.ndarray, np.ndarray]
    ) -> dict:
    weights, bias = head
    train = groups["train"]
    functions = {
        BASELINE_NAME: make_max_softmax_scorer(temperature),
        "max_logit": score_max_logit,
        "energy": score_energy
    }

    for dimension in VIM_DIMENSIONS:
        vim_model = ood_scores.fit_vim(
            train["features"],
            train["logits"],
            weights,
            bias,
            dimension
        )
        functions[f"vim_{dimension}"] = make_vim_scorer(vim_model)

    return functions

def score_group(score_function, group: dict, mask: np.ndarray | None = None) -> np.ndarray:
    features = group["features"]
    logits = group["logits"]

    if mask is not None:
        features = features[mask]
        logits = logits[mask]

    return score_function(features, logits)

def evaluate_methods(groups: dict, functions: dict, fit_folders: set, test_folders: set) -> dict:
    results = {}
    known_group = groups["val"]

    for view_name in ("unseen_box", "unseen_whole"):
        view = groups[view_name]
        fit_mask = make_folder_mask(view["source_ids"], fit_folders)
        test_mask = make_folder_mask(view["source_ids"], test_folders)
        results[view_name] = {}

        for method_name, score_function in functions.items():
            known_scores = score_group(score_function, known_group)
            results[view_name][method_name] = {
                "fit": ood_scores.summarize_scores(
                    known_scores,
                    score_group(score_function, view, fit_mask)
                ),
                "test": ood_scores.summarize_scores(
                    known_scores,
                    score_group(score_function, view, test_mask)
                )
            }

    return results

def read_false_accept(summary: dict, coverage: float) -> float:
    return summary[ood_scores.FALSE_ACCEPT_KEY.format(coverage=coverage)]

def choose_best_method(results: dict) -> str:
    best_name = BASELINE_NAME
    best_value = float("inf")

    for method_name, summaries in results["unseen_box"].items():
        if method_name == BASELINE_NAME:
            continue

        values = []

        for coverage in SELECTION_COVERAGES:
            values.append(read_false_accept(summaries["fit"], coverage))

        mean_value = float(np.mean(values))

        if mean_value < best_value:
            best_name = method_name
            best_value = mean_value

    return best_name

def decide_trigger(results: dict, chosen: str) -> dict:
    decision = {"chosen_on_fit_folders": chosen, "views": {}}

    for view_name in ("unseen_box", "unseen_whole"):
        baseline = read_false_accept(results[view_name][BASELINE_NAME]["test"], TRIGGER_COVERAGE)
        candidate = read_false_accept(results[view_name][chosen]["test"], TRIGGER_COVERAGE)
        decision["views"][view_name] = {
            "baseline_false_accept": baseline,
            "chosen_false_accept": candidate,
            "reduction": baseline - candidate,
            "passes_trigger": baseline - candidate >= TRIGGER_MARGIN
        }

    return decision

def report_lookalikes(groups: dict, functions: dict) -> dict:
    view = groups["unseen_box"]
    known_scores = {}
    report = {}

    for method_name, score_function in functions.items():
        known_scores[method_name] = score_group(score_function, groups["val"])

    for source_id in LOOKALIKE_SOURCE_IDS:
        mask = view["source_ids"] == source_id
        report[source_id] = {"count": int(mask.sum())}

        for method_name, score_function in functions.items():
            unseen_scores = score_group(score_function, view, mask)
            threshold = ood_scores.find_threshold_for_coverage(
                known_scores[method_name],
                TRIGGER_COVERAGE
            )
            false_accept = ood_scores.calculate_false_accept_rate(unseen_scores, threshold)
            report[source_id][method_name] = false_accept

    return report

def run_ood(data_root: Path) -> list[str]:
    groups = load_or_extract_groups(data_root)
    unseen_ids = list(groups["unseen_whole"]["source_ids"])
    unseen_ids.extend(groups["unseen_box"]["source_ids"])
    fit_folders, test_folders = split_unseen_folders(unseen_ids)
    head = read_classifier_head(data_root)
    functions = build_score_functions(groups, read_temperature(data_root), head)
    results = evaluate_methods(groups, functions, fit_folders, test_folders)
    chosen = choose_best_method(results)
    report = {
        "data_slice": "web_val",
        "known_count": int(len(groups["val"]["labels"])),
        "fit_folder_count": len(fit_folders),
        "test_folder_count": len(test_folders),
        "methods": results,
        "trigger": decide_trigger(results, chosen),
        "lookalikes_all_images": report_lookalikes(groups, functions)
    }
    output_path = data_root / MODEL_DIR_NAME / REPORT_NAME
    output_path.write_text(json.dumps(report, indent=JSON_INDENT), encoding="utf-8")

    return [
        f"trigger {report['trigger']}",
        f"wrote {output_path}"
    ]

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye stage2 ood")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    arguments = parser.parse_args(argv)
    start_run("stage2-ood", {"vim_dimensions": VIM_DIMENSIONS})

    return run_ood(arguments.data_root)
