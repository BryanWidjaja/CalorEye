import argparse
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from caloreye.data.export_s1 import DATASETS_DIR_NAME, STAGE_ONE_DIR_NAME, YAML_NAME
from caloreye.data.manifest import ACQUISITION_DIR_NAME
from caloreye.data.sam3_label import INSTANCES_NAME, STORE_DIR_NAME
from caloreye.paths import get_data_root_dir, get_runs_dir
from caloreye.runs import start_run

BASE_WEIGHTS_NAME = "yolo26n-seg.pt"
MODEL_DIR_NAME = "models"
TRAINED_WEIGHTS_NAME = "yolo26n-seg-food-trained.pt"
METRICS_NAME = "stage1_metrics.json"
SEED = 0
IMAGE_SIZE = 640
COUNT_CONFIDENCE = 0.25
COUNT_BATCH_SIZE = 16
BYTES_PER_GIB = 1024 ** 3
JSON_INDENT = 2
MIN_FREE_BYTES = 5 * BYTES_PER_GIB
SMOKE_EPOCHS = 2
SMOKE_FRACTION = 0.15
FULL_EPOCHS = 40
FULL_PATIENCE = 10
CLOSE_MOSAIC_EPOCHS = 10
DEFAULT_BATCH = 16
DEFAULT_WORKERS = 4
VAL_LIST_NAME = "val.txt"

@dataclass
class TrainSettings:
    is_smoke: bool
    batch: int
    workers: int
    epochs: int
    patience: int
    fraction: float
    plots: bool
    deterministic: bool

def build_settings(is_smoke: bool, batch: int, workers: int, epochs: int | None) -> TrainSettings:
    if is_smoke:
        return TrainSettings(
            is_smoke=True,
            batch=batch,
            workers=workers,
            epochs=SMOKE_EPOCHS,
            patience=SMOKE_EPOCHS,
            fraction=SMOKE_FRACTION,
            plots=False,
            deterministic=False
        )

    return TrainSettings(
        is_smoke=False,
        batch=batch,
        workers=workers,
        epochs=epochs or FULL_EPOCHS,
        patience=FULL_PATIENCE,
        fraction=1.0,
        plots=True,
        deterministic=True
    )

def build_train_arguments(data_yaml: Path, run_dir: Path, settings: TrainSettings) -> dict:
    return {
        "data": str(data_yaml),
        "epochs": settings.epochs,
        "patience": settings.patience,
        "imgsz": IMAGE_SIZE,
        "batch": settings.batch,
        "workers": settings.workers,
        "close_mosaic": CLOSE_MOSAIC_EPOCHS,
        "amp": True,
        "seed": SEED,
        "deterministic": settings.deterministic,
        "fraction": settings.fraction,
        "plots": settings.plots,
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,
        "device": 0
    }

def count_label_instances(data_root: Path) -> dict[str, int]:
    instances_path = data_root / STORE_DIR_NAME / INSTANCES_NAME
    payload = json.loads(instances_path.read_text(encoding="utf-8"))
    counts_by_id = {}

    for annotation in payload["annotations"]:
        image_id = annotation["image_id"]
        counts_by_id[image_id] = counts_by_id.get(image_id, 0) + 1

    counts_by_path = {}

    for record in payload["images"]:
        image_path = (data_root / ACQUISITION_DIR_NAME / record["path"]).as_posix()
        counts_by_path[image_path] = counts_by_id.get(record["image_id"], 0)

    return counts_by_path

def evaluate_counts(weights_path: Path, data_root: Path) -> dict:
    dataset_dir = data_root / DATASETS_DIR_NAME / STAGE_ONE_DIR_NAME
    val_paths = (dataset_dir / VAL_LIST_NAME).read_text(encoding="utf-8").split()
    label_counts = count_label_instances(data_root)
    model = YOLO(str(weights_path))
    differences = []
    predicted_counts = []

    for start in range(0, len(val_paths), COUNT_BATCH_SIZE):
        batch_paths = val_paths[start:start + COUNT_BATCH_SIZE]
        results = model.predict(
            batch_paths,
            imgsz=IMAGE_SIZE,
            conf=COUNT_CONFIDENCE,
            verbose=False
        )

        for image_path, result in zip(batch_paths, results, strict=True):
            predicted = len(result.boxes)
            predicted_counts.append(predicted)
            differences.append(predicted - label_counts[image_path])

    difference_array = np.array(differences)
    predicted_array = np.array(predicted_counts)

    return {
        "data_slice": "web_val",
        "label_source": "sam3 pseudo labels, not ground truth",
        "images": len(val_paths),
        "mean_predicted": float(np.mean(predicted_array)),
        "mean_labelled": float(np.mean(predicted_array - difference_array)),
        "exact_count_share": float(np.mean(difference_array == 0)),
        "within_one_share": float(np.mean(np.abs(difference_array) <= 1)),
        "mean_absolute_error": float(np.mean(np.abs(difference_array)))
    }

def collect_metrics(results) -> dict:
    metrics = {}

    for key, value in results.results_dict.items():
        metrics[key] = float(value)

    return metrics

def run_training(data_root: Path, settings: TrainSettings, resume_dir: Path | None) -> list[str]:
    free_bytes = shutil.disk_usage(data_root).free

    if free_bytes < MIN_FREE_BYTES:
        raise OSError(f"expected at least {MIN_FREE_BYTES} free bytes, found {free_bytes}")

    data_yaml = data_root / DATASETS_DIR_NAME / STAGE_ONE_DIR_NAME / YAML_NAME
    config = {"seed": SEED, "settings": asdict(settings), "data": str(data_yaml)}
    run_dir = start_run("stage1-train", config).run_dir
    arguments = build_train_arguments(data_yaml, run_dir, settings)
    base_weights = data_root / MODEL_DIR_NAME / BASE_WEIGHTS_NAME

    if resume_dir is not None:
        model = YOLO(str(resume_dir / "weights" / "last.pt"))
        arguments = {"resume": True}
    else:
        model = YOLO(str(base_weights))

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    results = model.train(**arguments)
    seconds = time.time() - started
    peak_gib = torch.cuda.max_memory_reserved() / BYTES_PER_GIB
    best_path = Path(model.trainer.best)
    summary = {
        "settings": asdict(settings),
        "wall_seconds": seconds,
        "peak_vram_reserved_gib": peak_gib,
        "metrics": collect_metrics(results),
        "best_weights": str(best_path),
        "data_slice": "web_val, agreement with sam3 pseudo labels"
    }

    if not settings.is_smoke:
        summary["counts"] = evaluate_counts(best_path, data_root)
        shutil.copyfile(best_path, data_root / MODEL_DIR_NAME / TRAINED_WEIGHTS_NAME)

    (run_dir / METRICS_NAME).write_text(json.dumps(summary, indent=JSON_INDENT), encoding="utf-8")
    logging.warning("stage one training finished in %.0f s", seconds)

    return [json.dumps(summary, indent=JSON_INDENT), f"run {run_dir}"]

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye stage1 train")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args(argv)
    get_runs_dir(create=True)
    settings = build_settings(arguments.smoke, arguments.batch, arguments.workers, arguments.epochs)

    return run_training(arguments.data_root, settings, arguments.resume)
