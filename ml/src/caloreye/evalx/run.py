import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from caloreye.data.export_s1 import (
    DATASETS_DIR_NAME,
    STAGE_ONE_DIR_NAME,
    decode_annotation_mask,
    group_annotations,
    load_instances,
)
from caloreye.data.manifest import ACQUISITION_DIR_NAME
from caloreye.evalx.kcal import PlateItem, load_nutrition, plate_errors, summarize_plates
from caloreye.evalx.match import match_instances, summarize_matches
from caloreye.evalx.metrics import (
    accuracy,
    confusion_matrix,
    expected_calibration_error,
    per_class_report,
)
from caloreye.paths import get_data_root_dir
from caloreye.runs import start_run
from caloreye.stage1.train import IMAGE_SIZE, MODEL_DIR_NAME
from caloreye.stage1.train import TRAINED_WEIGHTS_NAME as STAGE_ONE_WEIGHTS
from caloreye.stage2.calibrate import CALIBRATION_NAME, softmax_matrix
from caloreye.stage2.dataset import CropDataset, load_split_rows
from caloreye.stage2.export import load_trained_classifier
from caloreye.stage2.train import CLASS_NAMES, make_loader
from caloreye.stage2.train import TRAINED_WEIGHTS_NAME as STAGE_TWO_WEIGHTS

STAGE_ONE_CONFIDENCE = 0.25
REPORT_NAME = "eval.json"
JSON_INDENT = 2
SPLIT_VAL = "val"
DATA_SLICE_NOTE = (
    "web_val, not evidence for any project gate; "
    "own weighed-plate photos do not exist yet"
)


def read_stage_one_truth(data_root: Path) -> dict[str, list[np.ndarray]]:
    payload = load_instances(data_root)
    grouped = group_annotations(payload)
    truth = {}

    for record in payload["images"]:
        image_path = (data_root / ACQUISITION_DIR_NAME / record["path"]).as_posix()
        annotations = grouped.get(record["image_id"], [])
        truth[image_path] = [decode_annotation_mask(item) for item in annotations]

    return truth


def predict_stage_one_masks(model: YOLO, image_path: str, target_shape: tuple) -> np.ndarray:
    result = model.predict(
        image_path,
        imgsz=IMAGE_SIZE,
        conf=STAGE_ONE_CONFIDENCE,
        retina_masks=True,
        verbose=False
    )[0]

    if result.masks is None:
        return np.zeros((0, *target_shape), dtype=bool)

    resized = []

    for mask in result.masks.data.cpu().numpy():
        size = (target_shape[1], target_shape[0])
        resized.append(cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST))

    return np.stack(resized).astype(bool)


def evaluate_stage_one(data_root: Path) -> dict:
    list_path = data_root / DATASETS_DIR_NAME / STAGE_ONE_DIR_NAME / "val.txt"
    image_paths = list_path.read_text(encoding="utf-8").split()
    truth = read_stage_one_truth(data_root)
    model = YOLO(str(data_root / MODEL_DIR_NAME / STAGE_ONE_WEIGHTS))
    results = []

    for image_path in image_paths:
        truth_masks = np.stack(truth[image_path])
        predicted = predict_stage_one_masks(model, image_path, truth_masks.shape[1:])
        results.append(match_instances(predicted, truth_masks))

    summary = summarize_matches(results)
    summary["label_source"] = "sam3 pseudo labels, not ground truth"

    return summary


def evaluate_stage_two(data_root: Path) -> dict:
    rows = load_split_rows(data_root, SPLIT_VAL)
    calibration_path = data_root / MODEL_DIR_NAME / CALIBRATION_NAME
    temperature = json.loads(calibration_path.read_text(encoding="utf-8"))["temperature"]
    model = load_trained_classifier(data_root / MODEL_DIR_NAME / STAGE_TWO_WEIGHTS).cuda()
    loader = make_loader(CropDataset(data_root, rows, False, 0), False, None)
    logit_parts = []
    label_parts = []

    with torch.inference_mode():
        for images, labels, _ in loader:
            logit_parts.append(model(images.cuda()).cpu().numpy())
            label_parts.append(labels.numpy())

    logits = np.concatenate(logit_parts)
    labels = np.concatenate(label_parts)
    probabilities = softmax_matrix(logits, temperature)
    matrix = confusion_matrix(labels, probabilities.argmax(axis=1), len(CLASS_NAMES))
    ece, _ = expected_calibration_error(probabilities, labels)

    return {
        "count": len(rows),
        "top1": accuracy(matrix),
        "ece_calibrated": ece,
        "per_class": per_class_report(matrix, CLASS_NAMES),
        "confusion": matrix.tolist()
    }


def evaluate_plates(plates_path: Path) -> dict:
    table = load_nutrition()
    payload = json.loads(plates_path.read_text(encoding="utf-8"))
    plates = []

    for plate in payload:
        truth_items = [PlateItem(item["class"], item["grams"]) for item in plate["truth"]]
        matches = [(pair[0], pair[1]) for pair in plate["matches"]]
        plates.append(plate_errors(table, truth_items, plate["predicted"], matches))

    return summarize_plates(plates)


def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye eval run")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    parser.add_argument("--plates", type=Path)
    arguments = parser.parse_args(argv)
    run_dir = start_run("eval-run", {"plates": str(arguments.plates)}).run_dir
    report = {
        "data_slice": DATA_SLICE_NOTE,
        "stage_one": evaluate_stage_one(arguments.data_root),
        "stage_two": evaluate_stage_two(arguments.data_root)
    }

    if arguments.plates is not None:
        report["kcal"] = evaluate_plates(arguments.plates)
    else:
        report["kcal"] = "skipped: no plates file, own weighed plates do not exist yet"

    report_path = run_dir / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=JSON_INDENT), encoding="utf-8")
    stage_one = report["stage_one"]

    stage_two = report["stage_two"]

    return [
        f"stage one recall {stage_one['recall']:.3f} precision {stage_one['precision']:.3f} "
        f"merges {stage_one['merges']} splits {stage_one['splits']} (vs sam3 labels)",
        f"stage two top1 {stage_two['top1']:.3f} ece {stage_two['ece_calibrated']:.3f}",
        f"report {report_path}"
    ]
