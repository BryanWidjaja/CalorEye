import argparse
import csv
import json
import logging
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from caloreye.data.export_s2 import CLASS_SOURCE_IDS, NOT_FOOD_INDEX, ORIGIN_GT_BOX, ORIGIN_WHOLE
from caloreye.data.manifest import WORK_DIR_NAME
from caloreye.evalx.metrics import accuracy, confusion_matrix, macro_f1, per_class_report
from caloreye.export.bundle import CLASS_ROWS
from caloreye.paths import get_data_root_dir
from caloreye.runs import set_seed, start_run
from caloreye.stage2.dataset import (
    LOOKALIKE_SOURCE_IDS,
    SPLIT_TRAIN,
    SPLIT_VAL,
    CropDataset,
    build_cache,
    build_uncleaned_rows,
    build_unseen_rows,
    load_split_rows,
    sampler_weights,
)

MODEL_NAME = "efficientnet_b0.ra4_e3600_r224_in1k"
MODEL_DIR_NAME = "models"
TRAINED_WEIGHTS_NAME = "efficientnet-b0-seven-trained.pt"
CLASS_NAMES = [name for _, name in CLASS_ROWS]
CLASS_COUNT = len(CLASS_NAMES)
SEED = 0
BATCH_SIZE = 64
WORKER_COUNT = 2
HEAD_EPOCHS = 2
FULL_EPOCHS = 14
HEAD_LEARNING_RATE = 1e-3
BACKBONE_LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.02
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 1
TOP_LOSS_COUNT = 20
CONFIDENCE_CUTS = (0.5, 0.7, 0.9)
QUANTILES = (0.1, 0.5, 0.9)
GEPREK_INDEX = CLASS_SOURCE_IDS.index("6")
FRIED_CHICKEN_INDEX = CLASS_SOURCE_IDS.index("5")
VARIANT_CLEANED = "cleaned"
VARIANT_UNCLEANED = "uncleaned"
METRICS_NAME = "stage2_metrics.json"
BEST_NAME = "best.pt"
LAST_NAME = "last.pt"
JSON_INDENT = 2
EVAL_BATCH_SIZE = 128

@dataclass
class TrainState:
    optimizer: torch.optim.Optimizer
    scheduler: object
    scaler: torch.amp.GradScaler

@dataclass
class RunSetup:
    data_root: Path
    run_dir: Path
    train_rows: list
    val_rows: list
    unseen_rows: list
    publish: bool

def create_model() -> torch.nn.Module:
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=CLASS_COUNT)

    return model.cuda()

def make_loader(dataset: CropDataset, is_train: bool, weights: list[float] | None) -> DataLoader:
    if is_train:
        generator = torch.Generator().manual_seed(SEED)
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), generator=generator)

        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=WORKER_COUNT,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )

    return DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=WORKER_COUNT,
        pin_memory=True
    )

def collect_logits(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logit_parts = []
    label_parts = []

    with torch.inference_mode():
        for images, labels, _ in loader:
            logit_parts.append(model(images.cuda(non_blocking=True)).cpu().numpy())
            label_parts.append(labels.numpy())

    return np.concatenate(logit_parts), np.concatenate(label_parts)

def set_phase_mode(model: torch.nn.Module, is_head_only: bool) -> None:
    model.train()

    if is_head_only:
        model.eval()
        model.get_classifier().train()

def train_one_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        state: TrainState,
        is_head_only: bool
    ) -> float:
    set_phase_mode(model, is_head_only)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    total_loss = 0.0
    batch_count = 0

    for images, labels, _ in loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        state.optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.float16):
            loss = criterion(model(images), labels)

        state.scaler.scale(loss).backward()
        state.scaler.step(state.optimizer)
        state.scaler.update()

        if state.scheduler is not None:
            state.scheduler.step()

        total_loss += float(loss)
        batch_count += 1

    return total_loss / max(batch_count, 1)

def build_full_optimizer(model: torch.nn.Module, iterations_per_epoch: int):
    head_parameters = list(model.get_classifier().parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = []

    for parameter in model.parameters():
        parameter.requires_grad = True

        if id(parameter) not in head_ids:
            backbone_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": BACKBONE_LEARNING_RATE},
            {"params": head_parameters, "lr": HEAD_LEARNING_RATE}
        ],
        weight_decay=WEIGHT_DECAY
    )
    warmup_iterations = WARMUP_EPOCHS * iterations_per_epoch
    total_iterations = FULL_EPOCHS * iterations_per_epoch

    def schedule(iteration: int) -> float:
        if iteration < warmup_iterations:
            return (iteration + 1) / warmup_iterations

        progress = (iteration - warmup_iterations) / max(total_iterations - warmup_iterations, 1)

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

def build_head_optimizer(model: torch.nn.Module):
    for parameter in model.parameters():
        parameter.requires_grad = False

    head_parameters = list(model.get_classifier().parameters())

    for parameter in head_parameters:
        parameter.requires_grad = True

    return torch.optim.AdamW(head_parameters, lr=HEAD_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

def evaluate_split(logits: np.ndarray, labels: np.ndarray, rows: list[dict]) -> dict:
    predictions = logits.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, CLASS_COUNT)
    result = {
        "count": len(rows),
        "accuracy": accuracy(matrix),
        "macro_f1": macro_f1(matrix),
        "per_class": per_class_report(matrix, CLASS_NAMES),
        "confusion": matrix.tolist(),
        "geprek_as_fried_chicken": int(matrix[GEPREK_INDEX, FRIED_CHICKEN_INDEX]),
        "fried_chicken_as_geprek": int(matrix[FRIED_CHICKEN_INDEX, GEPREK_INDEX])
    }

    return result

def evaluate_by_origin(logits: np.ndarray, labels: np.ndarray, rows: list[dict]) -> dict:
    summary = {"combined": evaluate_split(logits, labels, rows)}

    for origin in (ORIGIN_WHOLE, ORIGIN_GT_BOX, "bukan_makanan"):
        positions = [index for index, row in enumerate(rows) if row["origin"] == origin]

        if not positions:
            continue

        subset_rows = [rows[index] for index in positions]
        summary[origin] = evaluate_split(logits[positions], labels[positions], subset_rows)

    return summary

def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)

    return exponentials / exponentials.sum(axis=1, keepdims=True)

def describe_unseen(logits: np.ndarray, rows: list[dict]) -> dict:
    probabilities = softmax_numpy(logits)
    maxima = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    sources = np.array([row["source_id"] for row in rows])
    groups = {"all_unseen": np.ones(len(rows), dtype=bool)}

    for source_id in LOOKALIKE_SOURCE_IDS:
        groups[f"lookalike_{source_id}"] = sources == source_id

    summary = {}

    for name, mask in groups.items():
        if not mask.any():
            continue

        group_max = maxima[mask]
        distribution = np.bincount(predictions[mask], minlength=CLASS_COUNT) / mask.sum()
        summary[name] = {
            "count": int(mask.sum()),
            "max_softmax_quantiles": {
                str(cut): float(np.quantile(group_max, cut)) for cut in QUANTILES
            },
            "share_above": {str(cut): float((group_max >= cut).mean()) for cut in CONFIDENCE_CUTS},
            "predicted_distribution": {
                CLASS_NAMES[index]: float(share) for index, share in enumerate(distribution)
            }
        }

    return summary

def write_top_loss(
        logits: np.ndarray,
        labels: np.ndarray,
        rows: list[dict],
        output_path: Path
    ) -> None:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(shifted).sum(axis=1))
    losses = log_norm - shifted[np.arange(len(labels)), labels]
    predictions = logits.argmax(axis=1)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["class", "image_id", "path", "origin", "loss", "predicted"])

        for class_index in range(CLASS_COUNT):
            positions = np.flatnonzero(labels == class_index)
            ordered = positions[np.argsort(-losses[positions])][:TOP_LOSS_COUNT]

            for position in ordered:
                row = rows[position]
                writer.writerow(
                    [
                        CLASS_NAMES[class_index],
                        row["image_id"],
                        row["path"],
                        row["origin"],
                        f"{losses[position]:.4f}",
                        CLASS_NAMES[predictions[position]]
                    ]
                )

def prepare_rows(data_root: Path, variant: str) -> tuple[list[dict], list[dict], list[dict]]:
    train_rows = load_split_rows(data_root, SPLIT_TRAIN)

    if variant == VARIANT_UNCLEANED:
        train_rows = train_rows + build_uncleaned_rows(data_root)

    val_rows = load_split_rows(data_root, SPLIT_VAL)
    unseen_rows = build_unseen_rows(data_root)
    build_cache(data_root, train_rows + val_rows + unseen_rows)

    return train_rows, val_rows, unseen_rows

def run_training(data_root: Path, variant: str, publish: bool) -> list[str]:
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    run_dir = start_run("stage2-train", {"variant": variant, "seed": SEED}).run_dir
    train_rows, val_rows, unseen_rows = prepare_rows(data_root, variant)
    train_dataset = CropDataset(data_root, train_rows, True, SEED)
    train_loader = make_loader(train_dataset, True, sampler_weights(train_rows, CLASS_COUNT))
    val_loader = make_loader(CropDataset(data_root, val_rows, False, SEED), False, None)
    model = create_model()
    scaler = torch.amp.GradScaler("cuda")
    setup = RunSetup(data_root, run_dir, train_rows, val_rows, unseen_rows, publish)
    history = []
    best_f1 = -1.0
    started = time.time()
    head_optimizer = build_head_optimizer(model)

    for epoch in range(HEAD_EPOCHS + FULL_EPOCHS):
        is_head_only = epoch < HEAD_EPOCHS

        if epoch == HEAD_EPOCHS:
            optimizer, scheduler = build_full_optimizer(model, len(train_loader))
            state = TrainState(optimizer, scheduler, scaler)

        if is_head_only:
            state = TrainState(head_optimizer, None, scaler)

        loss = train_one_epoch(model, train_loader, state, is_head_only)
        logits, labels = collect_logits(model, val_loader)
        matrix = confusion_matrix(labels, logits.argmax(axis=1), CLASS_COUNT)
        epoch_f1 = macro_f1(matrix)
        history.append({"epoch": epoch + 1, "train_loss": loss, "val_macro_f1": epoch_f1})
        logging.warning("epoch %d loss %.4f val macro f1 %.4f", epoch + 1, loss, epoch_f1)
        torch.save(model.state_dict(), run_dir / LAST_NAME)

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            torch.save(model.state_dict(), run_dir / BEST_NAME)

    return finish_run(model, setup, history, started)

def finish_run(
        model: torch.nn.Module,
        setup: RunSetup,
        history: list[dict],
        started: float
    ) -> list[str]:
    data_root = setup.data_root
    run_dir = setup.run_dir
    train_rows = setup.train_rows
    val_rows = setup.val_rows
    unseen_rows = setup.unseen_rows
    model.load_state_dict(torch.load(run_dir / BEST_NAME))
    val_logits, val_labels = collect_logits(
        model,
        make_loader(CropDataset(data_root, val_rows, False, SEED), False, None)
    )
    unseen_logits, _ = collect_logits(
        model,
        make_loader(CropDataset(data_root, unseen_rows, False, SEED), False, None)
    )
    train_logits, train_labels = collect_logits(
        model,
        make_loader(CropDataset(data_root, train_rows, False, SEED), False, None)
    )
    top_loss_path = data_root / WORK_DIR_NAME / f"s2_toploss-{run_dir.name}.csv"
    write_top_loss(train_logits, train_labels, train_rows, top_loss_path)
    report = {
        "data_slice": "web_val",
        "wall_seconds": time.time() - started,
        "history": history,
        "val": evaluate_by_origin(val_logits, val_labels, val_rows),
        "unseen": describe_unseen(unseen_logits, unseen_rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "top_loss_csv": str(top_loss_path),
        "not_food_index": NOT_FOOD_INDEX
    }
    (run_dir / METRICS_NAME).write_text(json.dumps(report, indent=JSON_INDENT), encoding="utf-8")

    if setup.publish:
        shutil.copyfile(run_dir / BEST_NAME, data_root / MODEL_DIR_NAME / TRAINED_WEIGHTS_NAME)

    combined = report["val"]["combined"]

    return [
        f"val combined accuracy {combined['accuracy']:.4f} macro f1 {combined['macro_f1']:.4f}",
        f"report {run_dir / METRICS_NAME}"
    ]

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye stage2 train")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    variants = [VARIANT_CLEANED, VARIANT_UNCLEANED]
    parser.add_argument("--variant", choices=variants, default=VARIANT_CLEANED)
    parser.add_argument("--publish", action="store_true")
    arguments = parser.parse_args(argv)

    return run_training(arguments.data_root, arguments.variant, arguments.publish)
