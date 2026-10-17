import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from caloreye.data.export_s1 import DATASETS_DIR_NAME, STAGE_ONE_DIR_NAME
from caloreye.export.parity import collect_stage_one_parity, compare_one_image
from caloreye.export.spike import export_stage_one
from caloreye.paths import get_data_root_dir
from caloreye.runs import start_run
from caloreye.stage1.train import MODEL_DIR_NAME, TRAINED_WEIGHTS_NAME, VAL_LIST_NAME

IMAGE_COUNT = 20
TRAINED_CLASS_COUNT = 1
REPORT_NAME = "stage1_trained_parity.json"
OVERLAY_NAME = "trained_decode_overlay.jpg"
JSON_INDENT = 2

def select_val_images(data_root: Path, image_count: int) -> list[Path]:
    list_path = data_root / DATASETS_DIR_NAME / STAGE_ONE_DIR_NAME / VAL_LIST_NAME
    val_paths = sorted(list_path.read_text(encoding="utf-8").split())
    stride = max(len(val_paths) // image_count, 1)

    return [Path(path) for path in val_paths[::stride][:image_count]]

def compare_decoder(weights_path: Path, image_paths: list[Path], run_dir: Path) -> dict:
    model = YOLO(str(weights_path))
    model.model.eval()
    model.model.model[-1].end2end = False
    comparisons = []

    for position, image_path in enumerate(image_paths):
        overlay_path = None

        if position == 0:
            overlay_path = run_dir / OVERLAY_NAME

        comparisons.append(compare_one_image(model, image_path, overlay_path, TRAINED_CLASS_COUNT))

    return {
        "data_slice": "web_val",
        "model": "trained 1-class yolo26n-seg",
        "images": comparisons,
        "matched_instances": sum(item["matched_count"] for item in comparisons),
        "passed": all(item["passed"] for item in comparisons)
    }

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye stage1 verify")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    parser.add_argument("--count", type=int, default=IMAGE_COUNT)
    arguments = parser.parse_args(argv)
    run_dir = start_run("stage1-verify", {"count": arguments.count}).run_dir
    weights_path = arguments.data_root / MODEL_DIR_NAME / TRAINED_WEIGHTS_NAME
    image_paths = select_val_images(arguments.data_root, arguments.count)
    program_path = export_stage_one(weights_path, "segment")
    parity = collect_stage_one_parity(TRAINED_WEIGHTS_NAME, image_paths)
    comparison = compare_decoder(weights_path, image_paths, run_dir)
    report = {"program": str(program_path), "parity": parity, "decode_comparison": comparison}
    report_path = run_dir / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=JSON_INDENT), encoding="utf-8")

    return [
        f"detection parity passed {parity['detection']['passed']}",
        f"prototype parity passed {parity['prototypes']['passed']}",
        f"decoder matched instances {comparison['matched_instances']}, passed {comparison['passed']}",
        f"report {report_path}"
    ]
