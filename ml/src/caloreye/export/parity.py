import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from executorch.runtime import Runtime
from PIL import Image, ImageDraw
from ultralytics import YOLO

from caloreye.paths import get_data_root_dir
from caloreye.stage1 import decode

MODEL_DIR_NAME = "models"
ACQUISITION_RELATIVE_PATH = Path("acquisition") / "images"
SOURCE_IDS = ("6", "5", "39", "B1", "11", "1")
CROWDED_SOURCE_IDS = ("3", "4", "B92", "B93", "B95")
CROWDED_IMAGE_COUNT = 10
MAX_IMAGE_ROUNDS = 4
PARITY_IMAGE_COUNT = 20
PREDICTION_IMAGE_COUNT = 10
ABSOLUTE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = 1e-3
COCO_CLASS_COUNT = 80
STAGE_TWO_SIZE = 224
STAGE_TWO_CLASS_COUNT = 7
STAGE_TWO_PADDING = 128
PIXEL_SCALE = 255
STAGE_TWO_MEAN = 0.5
STAGE_TWO_STD = 0.5
NANOSECONDS_PER_MILLISECOND = 1_000_000
COCO_WEIGHTS_NAME = "yolo26n-seg.pt"
FOOD_WEIGHTS_NAME = "yolo26n-seg-food.pt"
CLASSIFIER_WEIGHTS_NAME = "efficientnet-b0-seven.pt"
CLASSIFIER_PROGRAM_NAME = "efficientnet-b0-seven.pte"
PARITY_REPORT_NAME = "parity.json"
COMPARISON_REPORT_NAME = "decode_comparison.json"
DELEGATION_REPORT_NAME = "delegation_b0.json"
STAGE_ONE_DELEGATION_TEMPLATE = "delegation_{model_name}.json"
OVERLAY_IMAGE_NAME = "decode_overlay.jpg"
BOX_ERROR_LIMIT_PX = 1.0
MASK_IOU_LIMIT = 0.95
MATCH_IOU_LIMIT = 0.5
JSON_INDENT = 2

@dataclass
class DifferenceStats:
    maximum: float = 0.0
    total: float = 0.0
    value_count: int = 0
    violation_count: int = 0

    def add(self, reference: torch.Tensor, actual: torch.Tensor) -> None:
        if reference.shape != actual.shape:
            raise ValueError(
                f"Expected matching output shapes, found {reference.shape} and {actual.shape}"
            )

        difference = (reference.float() - actual.float()).abs()
        allowed_difference = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * reference.float().abs()
        self.violation_count += int((difference > allowed_difference).sum())
        self.maximum = max(self.maximum, float(difference.max()))
        self.total += float(difference.sum())
        self.value_count += difference.numel()

    def summary(self) -> dict:
        return {
            "max_abs": self.maximum,
            "mean_abs": self.total / self.value_count,
            "value_count": self.value_count,
            "tolerance_violations": self.violation_count,
            "passed": self.violation_count == 0
        }

def require_contiguous(input_tensor: torch.Tensor) -> None:
    if not input_tensor.is_contiguous():
        raise ValueError("ExecuTorch input must be contiguous")

def select_real_images(image_count: int, source_ids: tuple = SOURCE_IDS) -> list[Path]:
    acquisition_dir = get_data_root_dir() / ACQUISITION_RELATIVE_PATH
    image_paths = []
    source_groups = {}

    for source_id in source_ids:
        source_dir = acquisition_dir / source_id
        source_groups[source_id] = sorted(source_dir.glob("*"))

    for image_offset in range(MAX_IMAGE_ROUNDS):
        for source_id in source_ids:
            source_images = source_groups[source_id]

            if len(source_images) > image_offset:
                image_paths.append(source_images[image_offset])

        if len(image_paths) >= image_count:
            break

    if len(image_paths) < image_count:
        raise FileNotFoundError(f"Expected {image_count} real acquisition images")

    return image_paths[:image_count]

def prepare_stage_two_input(image_rgb: np.ndarray) -> torch.Tensor:
    original_height, original_width = image_rgb.shape[:2]
    square_size = max(original_height, original_width)
    top_padding = (square_size - original_height) // 2
    left_padding = (square_size - original_width) // 2
    square_image = np.full((square_size, square_size, 3), STAGE_TWO_PADDING, dtype=np.uint8)
    square_image[
        top_padding:top_padding + original_height,
        left_padding:left_padding + original_width
    ] = image_rgb
    resized_image = cv2.resize(
        square_image,
        (STAGE_TWO_SIZE, STAGE_TWO_SIZE),
        interpolation=cv2.INTER_LINEAR
    )
    channels_first = np.ascontiguousarray(resized_image.transpose(2, 0, 1))
    input_tensor = torch.from_numpy(channels_first).float().div(PIXEL_SCALE)
    input_tensor = (input_tensor - STAGE_TWO_MEAN) / STAGE_TWO_STD

    return input_tensor.unsqueeze(0).contiguous()

def load_runtime_method(program_path: Path):
    return Runtime.get().load_program(str(program_path)).load_method("forward")

def time_forward(forward_function, input_tensor: torch.Tensor) -> tuple:
    started_ns = time.perf_counter_ns()
    outputs = forward_function(input_tensor)
    duration_ms = (time.perf_counter_ns() - started_ns) / NANOSECONDS_PER_MILLISECOND

    return outputs, duration_ms

def time_runtime_forward(runtime_method, input_tensor: torch.Tensor) -> tuple:
    require_contiguous(input_tensor)
    started_ns = time.perf_counter_ns()
    outputs = runtime_method.execute([input_tensor])
    duration_ms = (time.perf_counter_ns() - started_ns) / NANOSECONDS_PER_MILLISECOND

    return outputs, duration_ms

def collect_stage_one_parity(model_name: str, image_paths: list[Path]) -> dict:
    model_dir = get_data_root_dir() / MODEL_DIR_NAME
    weights_path = model_dir / model_name
    program_path = weights_path.with_name(weights_path.stem + "_executorch_model") / "model.pte"
    model = YOLO(str(weights_path))
    model.model.eval()
    model.model.model[-1].end2end = False
    runtime_method = load_runtime_method(program_path)
    detection_differences = DifferenceStats()
    prototype_differences = DifferenceStats()
    pytorch_times_ms = []
    executorch_times_ms = []

    for image_path in image_paths:
        image_rgb = decode.load_image_rgb(image_path)
        input_tensor, _ = decode.prepare_stage_one_input(image_rgb)
        require_contiguous(input_tensor)

        with torch.inference_mode():
            pytorch_prediction, pytorch_ms = time_forward(model.model, input_tensor)

        pytorch_output = pytorch_prediction[0]
        runtime_output, executorch_ms = time_runtime_forward(runtime_method, input_tensor)
        detection_differences.add(pytorch_output[0], runtime_output[0])
        prototype_differences.add(pytorch_output[1], runtime_output[1])
        pytorch_times_ms.append(pytorch_ms)
        executorch_times_ms.append(executorch_ms)

    return {
        "detection": detection_differences.summary(),
        "prototypes": prototype_differences.summary(),
        "pytorch_host_cpu_ms_mean": float(np.mean(pytorch_times_ms)),
        "executorch_host_cpu_ms_mean": float(np.mean(executorch_times_ms)),
        "program_bytes": program_path.stat().st_size,
        "image_count": len(image_paths)
    }

def collect_stage_two_parity(image_paths: list[Path]) -> dict:
    model_dir = get_data_root_dir() / MODEL_DIR_NAME
    classifier = timm.create_model(
        "efficientnet_b0.ra4_e3600_r224_in1k",
        pretrained=False,
        num_classes=STAGE_TWO_CLASS_COUNT
    ).eval()
    weights_path = model_dir / CLASSIFIER_WEIGHTS_NAME
    classifier.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    program_path = model_dir / CLASSIFIER_PROGRAM_NAME
    runtime_method = load_runtime_method(program_path)
    logit_differences = DifferenceStats()
    pytorch_times_ms = []
    executorch_times_ms = []

    for image_path in image_paths:
        input_tensor = prepare_stage_two_input(decode.load_image_rgb(image_path))
        require_contiguous(input_tensor)

        with torch.inference_mode():
            pytorch_output, pytorch_ms = time_forward(classifier, input_tensor)

        runtime_output, executorch_ms = time_runtime_forward(runtime_method, input_tensor)
        logit_differences.add(pytorch_output, runtime_output[0])
        pytorch_times_ms.append(pytorch_ms)
        executorch_times_ms.append(executorch_ms)

    return {
        "logits": logit_differences.summary(),
        "pytorch_host_cpu_ms_mean": float(np.mean(pytorch_times_ms)),
        "executorch_host_cpu_ms_mean": float(np.mean(executorch_times_ms)),
        "program_bytes": program_path.stat().st_size,
        "image_count": len(image_paths)
    }

def calculate_mask_iou(first_mask: np.ndarray, second_mask: np.ndarray) -> float:
    intersection = np.logical_and(first_mask, second_mask).sum()
    union = np.logical_or(first_mask, second_mask).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)

def save_box_overlay(image_rgb, decoded, prediction, params, overlay_path: Path) -> None:
    overlay = Image.fromarray(image_rgb)
    drawing = ImageDraw.Draw(overlay)

    for detection in decoded:
        drawing.rectangle(detection.box_xyxy, outline=(0, 255, 0), width=2)

    for predicted_box in prediction.boxes.xyxy.cpu().numpy():
        original_box = decode.restore_original_box(predicted_box, params)
        drawing.rectangle(original_box, outline=(255, 0, 0), width=2)

    overlay.save(overlay_path)

def compare_one_image(
        model: YOLO,
        image_path: Path,
        overlay_path: Path | None,
        class_count: int = COCO_CLASS_COUNT
    ) -> dict:
    image_rgb = decode.load_image_rgb(image_path)
    input_tensor, params = decode.prepare_stage_one_input(image_rgb)
    canvas_rgb = input_tensor[0].permute(1, 2, 0).mul(PIXEL_SCALE).round().byte().numpy()
    canvas_bgr = np.ascontiguousarray(canvas_rgb[..., ::-1])

    with torch.inference_mode():
        raw_detections, raw_prototypes = model.model(input_tensor)[0]

    settings = decode.DecodeSettings(class_count=class_count, remove_contained=False)
    decoded = decode.decode_stage_one(raw_detections, raw_prototypes, params, settings)
    prediction = model.predict(
        canvas_bgr,
        imgsz=params.input_width_px,
        rect=False,
        conf=settings.score_threshold,
        iou=settings.nms_iou_threshold,
        agnostic_nms=True,
        verbose=False
    )[0]

    if overlay_path is not None:
        save_box_overlay(image_rgb, decoded, prediction, params, overlay_path)

    return compare_predictions(decoded, prediction, params, image_path)

def compare_predictions(decoded, prediction, params, image_path: Path) -> dict:
    predicted_boxes = prediction.boxes.xyxy.cpu().numpy()
    predicted_boxes_original = np.asarray(
        [decode.restore_original_box(predicted_box, params) for predicted_box in predicted_boxes]
    )
    predicted_classes = prediction.boxes.cls.cpu().numpy().astype(int)

    if prediction.masks is None:
        predicted_masks = np.empty((0, params.input_height_px, params.input_width_px))
    else:
        predicted_masks = prediction.masks.data.cpu().numpy()

    box_errors = []
    mask_ious = []
    unmatched_indices = set(range(len(predicted_boxes)))

    for detection in decoded:
        best_index = find_matching_prediction(
            detection,
            predicted_boxes_original,
            predicted_classes,
            unmatched_indices
        )

        if best_index is None:
            continue

        unmatched_indices.remove(best_index)
        predicted_box = predicted_boxes_original[best_index]
        predicted_mask = decode.restore_original_mask(
            predicted_masks[best_index].astype(np.uint8),
            params
        )
        box_errors.append(float(np.max(np.abs(np.asarray(detection.box_xyxy) - predicted_box))))
        mask_ious.append(calculate_mask_iou(detection.mask, predicted_mask))

    same_instances = len(decoded) == len(predicted_boxes) and not unmatched_indices
    passed = same_instances and all(error <= BOX_ERROR_LIMIT_PX for error in box_errors)
    passed = passed and all(mask_iou >= MASK_IOU_LIMIT for mask_iou in mask_ious)

    return {
        "image": image_path.name,
        "decoded_count": len(decoded),
        "ultralytics_count": len(predicted_boxes),
        "matched_count": len(box_errors),
        "max_box_error_px": max(box_errors, default=None),
        "min_mask_iou": min(mask_ious, default=None),
        "passed": passed
    }

def find_matching_prediction(detection, predicted_boxes, predicted_classes, unmatched_indices):
    best_index = None
    best_iou = MATCH_IOU_LIMIT

    for prediction_index in unmatched_indices:
        if predicted_classes[prediction_index] != detection.class_id:
            continue

        box_iou = decode.calculate_box_iou(
            np.asarray(detection.box_xyxy),
            predicted_boxes[prediction_index]
        )

        if box_iou > best_iou:
            best_index = prediction_index
            best_iou = box_iou

    return best_index

def collect_decode_comparison(image_paths: list[Path], run_dir: Path) -> dict:
    weights_path = get_data_root_dir() / MODEL_DIR_NAME / COCO_WEIGHTS_NAME
    model = YOLO(str(weights_path))
    model.model.eval()
    model.model.model[-1].end2end = False
    comparisons = []

    for image_index, image_path in enumerate(image_paths):
        overlay_path = None

        if image_index == 0:
            overlay_path = run_dir / OVERLAY_IMAGE_NAME

        comparisons.append(compare_one_image(model, image_path, overlay_path))

    return {
        "data_slice": "web_val",
        "model": "COCO pretrained yolo26n-seg",
        "images": comparisons,
        "total_matched_instances": sum(comparison["matched_count"] for comparison in comparisons),
        "most_instances_in_one_image": max(comparison["decoded_count"] for comparison in comparisons),
        "passed": all(comparison["passed"] for comparison in comparisons)
    }

def write_report(report_path: Path, report: dict) -> None:
    report_path.write_text(json.dumps(report, indent=JSON_INDENT) + "\n", encoding="utf-8")

def list_non_delegated_operators(report_path: Path) -> list[str]:
    delegation_report = json.loads(report_path.read_text(encoding="utf-8"))
    operator_names = []

    for operator_name, counts in delegation_report["operators"].items():
        if counts["non_delegated"] > 0:
            operator_names.append(operator_name)

    return operator_names

def collect_parity_verdict(parity_report: dict) -> dict:
    checks = {
        "coco_detection": parity_report["coco"]["detection"]["passed"],
        "coco_prototypes": parity_report["coco"]["prototypes"]["passed"],
        "food_detection": parity_report["food"]["detection"]["passed"],
        "food_prototypes": parity_report["food"]["prototypes"]["passed"],
        "classifier_logits": parity_report["classifier"]["logits"]["passed"]
    }

    return {
        "rule": "abs_error <= 1e-4 + 1e-3 * abs(reference), elementwise, initial estimate",
        "checks": checks,
        "passed": all(checks.values())
    }

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    image_paths = select_real_images(PARITY_IMAGE_COUNT)
    run_dir = arguments.run_dir.resolve()

    parity_report = {
        "data_slice": "web_val",
        "image_paths": [
            str(image_path.relative_to(get_data_root_dir()))
            for image_path in image_paths
        ],
        "coco": collect_stage_one_parity(COCO_WEIGHTS_NAME, image_paths),
        "food": collect_stage_one_parity(FOOD_WEIGHTS_NAME, image_paths),
        "classifier": collect_stage_two_parity(image_paths),
        "stage_two_non_delegated_operators": list_non_delegated_operators(
            run_dir / DELEGATION_REPORT_NAME
        ),
        "latency_scope": "host CPU, not a phone"
    }

    for model_name in ("coco", "food"):
        report_name = STAGE_ONE_DELEGATION_TEMPLATE.format(model_name=model_name)
        operator_names = list_non_delegated_operators(run_dir / report_name)
        parity_report[f"stage_one_{model_name}_non_delegated_operators"] = operator_names

    parity_report["verdict"] = collect_parity_verdict(parity_report)
    write_report(run_dir / PARITY_REPORT_NAME, parity_report)
    crowded_paths = select_real_images(CROWDED_IMAGE_COUNT, CROWDED_SOURCE_IDS)
    decode_paths = image_paths[:PREDICTION_IMAGE_COUNT] + crowded_paths
    decode_report = collect_decode_comparison(decode_paths, run_dir)
    write_report(run_dir / COMPARISON_REPORT_NAME, decode_report)

    if not decode_report["passed"]:
        raise RuntimeError("Reference decoder did not match Ultralytics on every checked image")

if __name__ == "__main__":
    main()
