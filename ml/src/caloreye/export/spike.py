import argparse
import json
import logging
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import timm
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import to_edge_transform_and_lower
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel

from caloreye.paths import get_data_root_dir
from caloreye.runs import set_seed, start_run

MODEL_DIR_NAME = "models"
COCO_WEIGHTS_NAME = "yolo26n-seg.pt"
FOOD_WEIGHTS_NAME = "yolo26n-seg-food.pt"
CLASSIFIER_WEIGHTS_NAME = "efficientnet-b0-seven.pt"
CLASSIFIER_PROGRAM_NAME = "efficientnet-b0-seven.pte"
DELEGATION_REPORT_NAME = "delegation_b0.json"
STAGE_ONE_DELEGATION_TEMPLATE = "delegation_{model_name}.json"
EXPORT_REPORT_NAME = "export_spike.json"
INPUT_SIZE_STAGE_ONE = 640
INPUT_SIZE_STAGE_TWO = 224
STAGE_TWO_CLASS_COUNT = 7
SEED = 0
MIN_FREE_DISK_BYTES = 1024 ** 3
JSON_INDENT = 2

def write_json(output_path: Path, value: dict) -> None:
    output_path.write_text(json.dumps(value, indent=JSON_INDENT) + "\n", encoding="utf-8")

def check_disk_space(data_dir: Path) -> None:
    free_bytes = shutil.disk_usage(data_dir).free
    logging.info("Free disk before export: %.2f GiB", free_bytes / MIN_FREE_DISK_BYTES)

    if free_bytes < MIN_FREE_DISK_BYTES:
        raise OSError("At least 1 GiB of free disk is required for the export spike")

def ensure_coco_weights(model_dir: Path) -> Path:
    weights_path = model_dir / COCO_WEIGHTS_NAME

    if weights_path.is_file():
        return weights_path

    pretrained = YOLO(COCO_WEIGHTS_NAME)
    shutil.copyfile(pretrained.ckpt_path, weights_path)

    return weights_path

def ensure_food_weights(model_dir: Path, coco_weights_path: Path) -> Path:
    weights_path = model_dir / FOOD_WEIGHTS_NAME

    if weights_path.is_file():
        existing_model = YOLO(str(weights_path))

        if existing_model.task == "segment":
            return weights_path

    pretrained = YOLO(str(coco_weights_path))
    one_class = SegmentationModel(deepcopy(pretrained.model.yaml), nc=1, verbose=False)
    one_class.names = {0: "food"}
    one_class.args = pretrained.model.args.copy()
    one_class.args["task"] = "segment"
    one_class.task = "segment"
    one_class.load(pretrained.model)
    one_class.pt_path = weights_path

    wrapper = YOLO("yolo26n-seg.yaml")
    wrapper.model = one_class
    wrapper.ckpt = {}
    wrapper.task = "segment"
    wrapper.save(weights_path)

    return weights_path

def export_stage_one(weights_path: Path, expected_task: str) -> Path:
    export_dir = weights_path.with_name(weights_path.stem + "_executorch_model")
    program_path = export_dir / "model.pte"
    metadata_path = export_dir / "metadata.yaml"

    if program_path.is_file() and metadata_path.is_file():
        metadata = metadata_path.read_text(encoding="utf-8")

        if f"task: {expected_task}" in metadata:
            return program_path

    model = YOLO(str(weights_path))
    exported_dir = Path(
        model.export(
            format="executorch",
            imgsz=INPUT_SIZE_STAGE_ONE,
            batch=1,
            device="cpu"
        )
    )
    program_path = exported_dir / "model.pte"

    if not program_path.is_file():
        raise FileNotFoundError(f"Expected exported model at {program_path}")

    return program_path

def collect_delegation_report(edge_program) -> dict:
    delegation = get_delegation_info(edge_program.exported_program().graph_module)
    operators = {}

    for operator_name, counts in delegation.delegation_by_operator.items():
        operators[operator_name] = {
            "delegated": counts.delegated,
            "non_delegated": counts.non_delegated
        }

    return {
        "delegated_subgraphs": delegation.num_delegated_subgraphs,
        "delegated_nodes": delegation.num_delegated_nodes,
        "non_delegated_nodes": delegation.num_non_delegated_nodes,
        "operators": operators
    }

def audit_stage_one_delegation(weights_path: Path, report_path: Path) -> None:
    if report_path.is_file():
        return

    captured_programs = []

    def record_lowering(*arguments, **keyword_arguments):
        edge_program = to_edge_transform_and_lower(*arguments, **keyword_arguments)
        captured_programs.append(edge_program)

        return edge_program

    with tempfile.TemporaryDirectory() as temporary_name:
        copied_weights_path = Path(temporary_name) / weights_path.name
        shutil.copyfile(weights_path, copied_weights_path)

        with patch("executorch.exir.to_edge_transform_and_lower", record_lowering):
            export_stage_one(copied_weights_path, "segment")

    if len(captured_programs) != 1:
        raise RuntimeError(f"Expected one lowering call, found {len(captured_programs)}")

    write_json(report_path, collect_delegation_report(captured_programs[0]))

def export_stage_two(model_dir: Path, run_dir: Path) -> Path:
    program_path = model_dir / CLASSIFIER_PROGRAM_NAME
    report_path = run_dir / DELEGATION_REPORT_NAME

    if program_path.is_file() and report_path.is_file():
        return program_path

    classifier = timm.create_model(
        "efficientnet_b0.ra4_e3600_r224_in1k",
        pretrained=True,
        num_classes=STAGE_TWO_CLASS_COUNT
    ).eval()
    example_input = torch.zeros(1, 3, INPUT_SIZE_STAGE_TWO, INPUT_SIZE_STAGE_TWO).contiguous()
    exported = torch.export.export(classifier, (example_input,))
    edge_program = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])

    write_json(report_path, collect_delegation_report(edge_program))
    torch.save(classifier.state_dict(), model_dir / CLASSIFIER_WEIGHTS_NAME)
    program_path.write_bytes(edge_program.to_executorch().buffer)

    return program_path

def inspect_stage_one_signature(weights_path: Path) -> dict:
    model = YOLO(str(weights_path))
    model.model.eval()
    model.model.model[-1].end2end = False
    example_input = torch.zeros(1, 3, INPUT_SIZE_STAGE_ONE, INPUT_SIZE_STAGE_ONE).contiguous()

    with torch.inference_mode():
        raw_outputs = model.model(example_input)[0]

    return {
        "detection_shape": list(raw_outputs[0].shape),
        "prototype_shape": list(raw_outputs[1].shape),
        "detection_dtype": str(raw_outputs[0].dtype),
        "prototype_dtype": str(raw_outputs[1].dtype)
    }

def run_export_spike(run_dir: Path) -> None:
    data_dir = get_data_root_dir()
    model_dir = data_dir / MODEL_DIR_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    check_disk_space(data_dir)
    set_seed(SEED)

    coco_weights_path = ensure_coco_weights(model_dir)
    food_weights_path = ensure_food_weights(model_dir, coco_weights_path)
    coco_program_path = export_stage_one(coco_weights_path, "segment")
    food_program_path = export_stage_one(food_weights_path, "segment")
    classifier_program_path = export_stage_two(model_dir, run_dir)

    for model_name, weights_path in (("coco", coco_weights_path), ("food", food_weights_path)):
        report_name = STAGE_ONE_DELEGATION_TEMPLATE.format(model_name=model_name)
        audit_stage_one_delegation(weights_path, run_dir / report_name)

    report = {
        "coco": inspect_stage_one_signature(coco_weights_path),
        "food": inspect_stage_one_signature(food_weights_path),
        "programs": {
            "coco": str(coco_program_path),
            "food": str(food_program_path),
            "classifier": str(classifier_program_path)
        },
        "bytes": {
            "coco": coco_program_path.stat().st_size,
            "food": food_program_path.stat().st_size,
            "classifier": classifier_program_path.stat().st_size
        }
    }
    write_json(run_dir / EXPORT_REPORT_NAME, report)
    logging.info("Export spike completed in %s", run_dir)

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if arguments.resume is None:
        run_dir = start_run("export-spike", {"seed": SEED}).run_dir
    else:
        run_dir = arguments.resume.resolve()

    run_export_spike(run_dir)

if __name__ == "__main__":
    main()
