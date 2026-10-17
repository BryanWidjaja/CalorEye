import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from caloreye.data.export_s1 import DATASETS_DIR_NAME, STAGE_ONE_DIR_NAME
from caloreye.data.manifest import LATEST_MANIFEST_NAME, WORK_DIR_NAME
from caloreye.export.bundle import (
    CLASS_ROWS,
    EXAMPLE_PATH,
    SCHEMA_PATH,
    describe_asset,
    describe_stage_one,
    describe_stage_two,
    validate_manifest,
)
from caloreye.export.parity import (
    calculate_mask_iou,
    collect_stage_one_parity,
    load_runtime_method,
    time_runtime_forward,
)
from caloreye.export.spike import (
    audit_stage_one_delegation,
    export_stage_one,
    inspect_stage_one_signature,
)
from caloreye.paths import get_data_root_dir
from caloreye.runs import hash_file, start_run
from caloreye.stage1 import decode
from caloreye.stage1.train import MODEL_DIR_NAME, TRAINED_WEIGHTS_NAME, VAL_LIST_NAME
from caloreye.stage2.calibrate import (
    CALIBRATION_NAME,
    PROGRAM_NAME,
    check_pytorch_parity,
)
from caloreye.stage2.dataset import CropDataset, load_split_rows
from caloreye.stage2.export import export_trained_classifier
from caloreye.stage2.train import TRAINED_WEIGHTS_NAME as STAGE_TWO_WEIGHTS

PARITY_IMAGE_COUNT = 50
FROZEN_SPLIT_NAME = "splits-frozen.json"
BUNDLE_NAME = "bundle_manifest.json"
REPORT_NAME = "trained_export.json"
STAGE_ONE_DELEGATION_NAME = "delegation_trained_stage1.json"
STAGE_TWO_DELEGATION_NAME = "delegation_trained_stage2.json"
JOINT_MODEL_VERSION = "trained-0"
NOT_FOOD_THRESHOLD = 0.5
STAGE_TWO_OUTPUT_SHAPE = [1, 7]
TIMING_RUNS = 50
BOX_ERROR_LIMIT_PX = 1.0
MASK_IOU_LIMIT = 0.95
TRAINED_CLASS_COUNT = 1
SPLIT_VAL = "val"
JSON_INDENT = 2


def select_val_images(data_root: Path, image_count: int) -> list[Path]:
    list_path = data_root / DATASETS_DIR_NAME / STAGE_ONE_DIR_NAME / VAL_LIST_NAME
    val_paths = sorted(list_path.read_text(encoding="utf-8").split())
    stride = max(len(val_paths) // image_count, 1)

    return [Path(path) for path in val_paths[::stride][:image_count]]


def time_stage_two_runtime(program_path: Path, data_root: Path, rows: list[dict]) -> dict:
    method = load_runtime_method(program_path)
    dataset = CropDataset(data_root, rows[:TIMING_RUNS], False, 0)
    durations = []
    output_shape = None

    for index in range(len(dataset)):
        tensor = dataset[index][0][None].contiguous()
        outputs, duration_ms = time_runtime_forward(method, tensor)
        durations.append(duration_ms)
        output_shape = list(outputs[0].shape)

    return {
        "runs": len(durations),
        "host_cpu_ms_mean": float(np.mean(durations)),
        "host_cpu_ms_p95": float(np.quantile(durations, 0.95)),
        "output_shape": output_shape,
        "scope": "host CPU, not a phone"
    }


def compare_runtime_decode(weights_path: Path, image_paths: list[Path]) -> dict:
    program_path = weights_path.with_name(weights_path.stem + "_executorch_model") / "model.pte"
    method = load_runtime_method(program_path)
    model = YOLO(str(weights_path))
    model.model.eval()
    model.model.model[-1].end2end = False
    settings = decode.DecodeSettings(class_count=TRAINED_CLASS_COUNT, remove_contained=False)
    mismatched_counts = 0
    box_errors = []
    mask_ious = []

    for image_path in image_paths:
        tensor, params = decode.prepare_stage_one_input(decode.load_image_rgb(image_path))

        with torch.inference_mode():
            reference = model.model(tensor)[0]

        runtime = method.execute([tensor.contiguous()])
        expected = decode.decode_stage_one(reference[0], reference[1], params, settings)
        actual = decode.decode_stage_one(runtime[0], runtime[1], params, settings)

        if len(expected) != len(actual):
            mismatched_counts += 1
            continue

        for first, second in zip(expected, actual, strict=True):
            difference = np.abs(np.asarray(first.box_xyxy) - np.asarray(second.box_xyxy))
            box_errors.append(float(difference.max()))
            mask_ious.append(calculate_mask_iou(first.mask, second.mask))

    return {
        "images": len(image_paths),
        "images_with_different_instance_count": mismatched_counts,
        "instances_compared": len(box_errors),
        "max_box_error_px": max(box_errors, default=0.0),
        "min_mask_iou": min(mask_ious, default=1.0),
        "passed": (
            mismatched_counts == 0
            and max(box_errors, default=0.0) <= BOX_ERROR_LIMIT_PX
            and min(mask_ious, default=1.0) >= MASK_IOU_LIMIT
        )
    }


def build_trained_manifest(data_root: Path, run_dir: Path, signature: dict) -> dict:
    model_dir = data_root / MODEL_DIR_NAME
    program_one = Path(f"{Path(TRAINED_WEIGHTS_NAME).stem}_executorch_model") / "model.pte"
    calibration = json.loads((model_dir / CALIBRATION_NAME).read_text(encoding="utf-8"))
    frozen_path = data_root / WORK_DIR_NAME / FROZEN_SPLIT_NAME
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    classes = []

    for class_index, (source_id, name_id) in enumerate(CLASS_ROWS):
        classes.append({"index": class_index, "source_id": source_id, "name_id": name_id})

    manifest = dict(example)
    manifest["bundle_id"] = run_record["run_id"]
    manifest["joint_model_version"] = JOINT_MODEL_VERSION
    manifest["created_at_utc"] = run_record["started_at_utc"]
    manifest["code_revision"] = run_record["git_sha"]
    dataset_path = data_root / WORK_DIR_NAME / LATEST_MANIFEST_NAME
    manifest["dataset_manifest_sha256"] = hash_file(dataset_path)
    manifest["split_sha256"] = frozen["split_hash"]
    manifest["assets"] = {
        "stage_one": describe_asset(model_dir, program_one),
        "stage_two": describe_asset(model_dir, Path(PROGRAM_NAME))
    }
    environment = dict(example["environment"])
    environment.update(run_record["versions"])
    environment["python"] = sys.version.split()[0]
    manifest["environment"] = environment
    manifest["stage_one"] = describe_stage_one(signature)
    manifest["stage_two"] = describe_stage_two()
    manifest["classes"] = classes
    manifest["thresholds"] = {
        "temperature": calibration["temperature"],
        "dish_acceptance": calibration["tau"]["tau_dish"],
        "not_food": NOT_FOOD_THRESHOLD
    }

    return manifest


def run_trained_export(data_root: Path) -> list[str]:
    run_dir = start_run("export-trained", {"parity_images": PARITY_IMAGE_COUNT}).run_dir
    model_dir = data_root / MODEL_DIR_NAME
    stage_one_weights = model_dir / TRAINED_WEIGHTS_NAME
    stage_two_weights = model_dir / STAGE_TWO_WEIGHTS
    stage_two_program = model_dir / PROGRAM_NAME
    program_one = export_stage_one(stage_one_weights, "segment")
    delegation_two = export_trained_classifier(stage_two_weights, stage_two_program)
    audit_stage_one_delegation(stage_one_weights, run_dir / STAGE_ONE_DELEGATION_NAME)
    image_paths = select_val_images(data_root, PARITY_IMAGE_COUNT)
    stage_one = collect_stage_one_parity(TRAINED_WEIGHTS_NAME, image_paths)
    val_rows = load_split_rows(data_root, SPLIT_VAL)
    stage_two = check_pytorch_parity(stage_two_weights, data_root, val_rows, stage_two_program)
    timing = time_stage_two_runtime(stage_two_program, data_root, val_rows)
    decode_report = compare_runtime_decode(stage_one_weights, image_paths)
    failures = collect_failures(decode_report, stage_two, timing)

    if failures:
        raise ValueError(f"expected parity and shapes to hold, found {failures}")

    signature = inspect_stage_one_signature(stage_one_weights)
    manifest = build_trained_manifest(data_root, run_dir, signature)
    validate_manifest(manifest, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")), model_dir)
    delegation_text = json.dumps(delegation_two, indent=JSON_INDENT)
    (run_dir / STAGE_TWO_DELEGATION_NAME).write_text(delegation_text, encoding="utf-8")
    report = {
        "data_slice": "web_val",
        "stage_one_raw_parity_strict": stage_one,
        "stage_one_decode_equivalence": decode_report,
        "stage_two": stage_two,
        "stage_two_timing": timing,
        "bytes": {
            "stage_one": program_one.stat().st_size,
            "stage_two": stage_two_program.stat().st_size
        }
    }
    (run_dir / REPORT_NAME).write_text(json.dumps(report, indent=JSON_INDENT), encoding="utf-8")
    bundle_text = json.dumps(manifest, indent=JSON_INDENT) + "\n"
    (model_dir / BUNDLE_NAME).write_text(bundle_text, encoding="utf-8")
    (run_dir / BUNDLE_NAME).write_text(bundle_text, encoding="utf-8")

    return summarize(report, run_dir, model_dir)


def collect_failures(decode_report: dict, stage_two: dict, timing: dict) -> list[str]:
    failures = []

    if not decode_report["passed"]:
        failures.append("stage one decoded outputs differ between pytorch and executorch")

    if not stage_two["passed"]:
        failures.append("stage two parity")

    if timing["output_shape"] != STAGE_TWO_OUTPUT_SHAPE:
        failures.append(f"stage two output shape {timing['output_shape']}")

    return failures


def summarize(report: dict, run_dir: Path, model_dir: Path) -> list[str]:
    stage_one = report["stage_one_raw_parity_strict"]
    stage_two = report["stage_two"]

    timing = report["stage_two_timing"]

    return [
        f"stage one strict raw parity detection {stage_one['detection']['passed']} "
        f"({stage_one['detection']['tolerance_violations']} violations) "
        f"prototypes {stage_one['prototypes']['passed']}, "
        f"decode equivalence {report['stage_one_decode_equivalence']['passed']}, "
        f"host cpu ms pytorch {stage_one['pytorch_host_cpu_ms_mean']:.1f} "
        f"executorch {stage_one['executorch_host_cpu_ms_mean']:.1f}",
        f"stage two parity {stage_two['passed']} max abs {stage_two['max_abs_difference']:.2e}, "
        f"host cpu ms {timing['host_cpu_ms_mean']:.1f}",
        f"bytes {report['bytes']}",
        f"bundle {model_dir / BUNDLE_NAME}",
        f"report {run_dir / REPORT_NAME}"
    ]


def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye export trained")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    arguments = parser.parse_args(argv)

    return run_trained_export(arguments.data_root)
