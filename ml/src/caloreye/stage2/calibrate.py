import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from caloreye.data.export_s2 import NOT_FOOD_INDEX
from caloreye.evalx.metrics import expected_calibration_error
from caloreye.export.parity import load_runtime_method, require_contiguous
from caloreye.paths import get_data_root_dir
from caloreye.runs import start_run
from caloreye.stage2.dataset import (
    LOOKALIKE_SOURCE_IDS,
    CropDataset,
    build_unseen_box_rows,
    build_unseen_rows,
    load_split_rows,
)
from caloreye.stage2.export import export_trained_classifier, load_trained_classifier
from caloreye.stage2.train import CLASS_NAMES, MODEL_DIR_NAME, TRAINED_WEIGHTS_NAME

PROGRAM_NAME = "efficientnet-b0-seven-trained.pte"
CALIBRATION_NAME = "efficientnet-b0-seven-trained.calibration.json"
COVERAGES = (1.0, 0.9, 0.75, 0.5)
TARGET_ACCURACY = 0.90
MIN_COVERAGE = 0.5
NOT_FOOD_THRESHOLDS = (0.3, 0.5, 0.7)
NOT_FOOD_REPORT_THRESHOLD = 0.5
LBFGS_ITERATIONS = 100
INITIAL_LOG_TEMPERATURE = 0.0
LOADER_WORKERS = 2
LOADER_BATCH = 64
PARITY_ROWS = 50
PARITY_ABS_TOLERANCE = 1e-4
PARITY_REL_TOLERANCE = 1e-3
PROBABILITY_FLOOR = 1e-12
JSON_INDENT = 2
SPLIT_VAL = "val"

def collect_runtime_logits(
        program_path: Path,
        data_root: Path,
        rows: list[dict]
    ) -> tuple[np.ndarray, np.ndarray]:
    method = load_runtime_method(program_path)
    loader = DataLoader(
        CropDataset(data_root, rows, False, 0),
        batch_size=LOADER_BATCH,
        num_workers=LOADER_WORKERS
    )
    logit_parts = []
    label_parts = []

    for images, labels, _ in loader:
        for position in range(images.shape[0]):
            single = images[position:position + 1].contiguous()
            require_contiguous(single)
            logit_parts.append(method.execute([single])[0].numpy().reshape(-1))

        label_parts.append(labels.numpy())

    return np.stack(logit_parts), np.concatenate(label_parts)

def check_pytorch_parity(
        weights_path: Path,
        data_root: Path,
        rows: list[dict],
        program_path: Path
    ) -> dict:
    subset = rows[:PARITY_ROWS]
    runtime_logits, _ = collect_runtime_logits(program_path, data_root, subset)
    model = load_trained_classifier(weights_path)
    dataset = CropDataset(data_root, subset, False, 0)
    reference = []

    with torch.inference_mode():
        for index in range(len(subset)):
            reference.append(model(dataset[index][0][None]).numpy().reshape(-1))

    reference_logits = np.stack(reference)
    difference = np.abs(reference_logits - runtime_logits)
    allowed = PARITY_ABS_TOLERANCE + PARITY_REL_TOLERANCE * np.abs(reference_logits)

    return {
        "rows": len(subset),
        "max_abs_difference": float(difference.max()),
        "violations": int((difference > allowed).sum()),
        "passed": bool((difference <= allowed).all())
    }

def softmax_matrix(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)

    return exponentials / exponentials.sum(axis=1, keepdims=True)

def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logit_tensor = torch.tensor(logits, dtype=torch.float64)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.tensor(INITIAL_LOG_TEMPERATURE, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=LBFGS_ITERATIONS)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled = logit_tensor / log_temperature.exp()
        loss = torch.nn.functional.cross_entropy(scaled, label_tensor)
        loss.backward()

        return loss

    optimizer.step(closure)

    return float(log_temperature.exp())

def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    chosen = probabilities[np.arange(len(labels)), labels]

    return float(-np.log(np.clip(chosen, PROBABILITY_FLOOR, 1.0)).mean())

def dish_confidence(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dish_probabilities = probabilities[:, :NOT_FOOD_INDEX]

    return dish_probabilities.max(axis=1), dish_probabilities.argmax(axis=1)

def coverage_curve(confidences: np.ndarray, correct: np.ndarray) -> list[dict]:
    order = np.argsort(-confidences)
    curve = []

    for coverage in COVERAGES:
        keep_count = max(int(round(coverage * len(order))), 1)
        kept = order[:keep_count]
        curve.append(
            {
                "coverage": coverage,
                "threshold": float(confidences[kept].min()),
                "accuracy": float(correct[kept].mean()),
                "count": keep_count
            }
        )

    return curve

def propose_tau(confidences: np.ndarray, correct: np.ndarray) -> dict:
    order = np.argsort(-confidences)
    sorted_correct = correct[order]
    cumulative = np.cumsum(sorted_correct) / np.arange(1, len(order) + 1)
    minimum_count = int(np.ceil(MIN_COVERAGE * len(order)))
    best = {"tau_dish": None, "coverage": None, "accuracy": None}

    for count in range(len(order), minimum_count - 1, -1):
        if cumulative[count - 1] >= TARGET_ACCURACY:
            best = {
                "tau_dish": float(confidences[order[count - 1]]),
                "coverage": count / len(order),
                "accuracy": float(cumulative[count - 1])
            }
            break

    return best

def not_food_report(probabilities: np.ndarray, labels: np.ndarray) -> list[dict]:
    is_not_food = labels == NOT_FOOD_INDEX
    report = []

    for threshold in NOT_FOOD_THRESHOLDS:
        flagged = probabilities[:, NOT_FOOD_INDEX] >= threshold
        true_positive = int((flagged & is_not_food).sum())
        precision = true_positive / max(int(flagged.sum()), 1)
        recall = true_positive / max(int(is_not_food.sum()), 1)
        report.append({"threshold": threshold, "precision": precision, "recall": recall})

    return report

def unseen_report(probabilities: np.ndarray, rows: list[dict], tau_dish: float | None) -> dict:
    confidences, _ = dish_confidence(probabilities)
    sources = np.array([row["source_id"] for row in rows])
    groups = {
        "all_unseen": np.ones(len(rows), dtype=bool),
        "lookalikes": np.isin(sources, LOOKALIKE_SOURCE_IDS)
    }
    report = {}

    for name, mask in groups.items():
        entry = {"count": int(mask.sum())}

        if tau_dish is not None:
            entry["false_accept_rate_at_tau"] = float((confidences[mask] >= tau_dish).mean())

        flagged = probabilities[mask, NOT_FOOD_INDEX] >= NOT_FOOD_REPORT_THRESHOLD
        entry["flagged_not_food"] = float(flagged.mean())
        report[name] = entry

    return report

def run_calibration(data_root: Path) -> list[str]:
    weights_path = data_root / MODEL_DIR_NAME / TRAINED_WEIGHTS_NAME
    program_path = data_root / MODEL_DIR_NAME / PROGRAM_NAME
    export_trained_classifier(weights_path, program_path)
    val_rows = load_split_rows(data_root, SPLIT_VAL)
    parity = check_pytorch_parity(weights_path, data_root, val_rows, program_path)

    if not parity["passed"]:
        raise ValueError(f"expected runtime logits to match PyTorch, found {parity}")

    logits, labels = collect_runtime_logits(program_path, data_root, val_rows)
    temperature = fit_temperature(logits, labels)
    raw_probabilities = softmax_matrix(logits)
    calibrated = softmax_matrix(logits, temperature)
    raw_ece, _ = expected_calibration_error(raw_probabilities, labels)
    ece, table = expected_calibration_error(calibrated, labels)
    dish_mask = labels != NOT_FOOD_INDEX
    confidences, predictions = dish_confidence(calibrated[dish_mask])
    correct = predictions == labels[dish_mask]
    tau = propose_tau(confidences, correct)
    unseen_rows = build_unseen_rows(data_root)
    unseen_logits, _ = collect_runtime_logits(program_path, data_root, unseen_rows)
    unseen_probabilities = softmax_matrix(unseen_logits, temperature)
    unseen_box_rows = build_unseen_box_rows(data_root)
    unseen_box_logits, _ = collect_runtime_logits(program_path, data_root, unseen_box_rows)
    unseen_box_probabilities = softmax_matrix(unseen_box_logits, temperature)
    result = {
        "data_slice": "web_val",
        "logits_source": "executorch runtime, fp32 xnnpack",
        "parity": parity,
        "temperature": temperature,
        "nll_before": negative_log_likelihood(raw_probabilities, labels),
        "nll_after": negative_log_likelihood(calibrated, labels),
        "ece_before": raw_ece,
        "ece_after": ece,
        "reliability_table": table,
        "coverage_curve": coverage_curve(confidences, correct),
        "tau": tau,
        "not_food": not_food_report(calibrated, labels),
        "unseen": unseen_report(unseen_probabilities, unseen_rows, tau["tau_dish"]),
        "unseen_boxes": unseen_report(
            unseen_box_probabilities,
            unseen_box_rows,
            tau["tau_dish"]
        ),
        "class_names": CLASS_NAMES,
        "not_food_note": "bukan_makanan has no real non-food photos, precision unvalidated"
    }
    output_path = data_root / MODEL_DIR_NAME / CALIBRATION_NAME
    output_path.write_text(json.dumps(result, indent=JSON_INDENT), encoding="utf-8")

    return [
        f"temperature {temperature:.3f}  ece {raw_ece:.4f} -> {ece:.4f}",
        f"tau_dish {tau}",
        f"unseen {result['unseen']}",
        f"unseen boxes {result['unseen_boxes']}",
        f"wrote {output_path}"
    ]

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye stage2 calibrate")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    arguments = parser.parse_args(argv)
    start_run("stage2-calibrate", {"seed": 0})

    return run_calibration(arguments.data_root)
