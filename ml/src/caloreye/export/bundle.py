import argparse
import json
import sys
from pathlib import Path

from caloreye.paths import REPO_ROOT_DIR, get_data_root_dir
from caloreye.runs import hash_file

CONTRACTS_DIR = REPO_ROOT_DIR / "ml" / "contracts"
SCHEMA_PATH = CONTRACTS_DIR / "bundle_manifest.schema.json"
EXAMPLE_PATH = CONTRACTS_DIR / "bundle_manifest.example.json"
MODEL_DIR_NAME = "models"
FOOD_PROGRAM_RELATIVE_PATH = Path("yolo26n-seg-food_executorch_model") / "model.pte"
CLASSIFIER_PROGRAM_RELATIVE_PATH = Path("efficientnet-b0-seven.pte")
RUN_RECORD_NAME = "run.json"
EXPORT_REPORT_NAME = "export_spike.json"
JSON_INDENT = 2
STAGE_ONE_INPUT_SIZE = 640
STAGE_TWO_INPUT_SIZE = 224
LETTERBOX_FILL = 114
STAGE_TWO_PADDING = 128
SCORE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 300
MASK_IOU_THRESHOLD = 0.8
CONTAINMENT_THRESHOLD = 0.9
CONTAINMENT_BOX_IOU = 0.5
EXPECTED_DETECTION_SHAPE = [1, 37, 8400]
EXPECTED_PROTOTYPE_SHAPE = [1, 32, 160, 160]
CLASS_ROWS = (
    ("6", "ayam geprek"),
    ("5", "ayam goreng"),
    ("39", "rendang"),
    ("B1", "beef burger"),
    ("11", "mie goreng"),
    ("1", "nasi putih"),
    ("bukan_makanan", "bukan_makanan")
)

def load_json(file_path: Path) -> dict:
    return json.loads(file_path.read_text(encoding="utf-8"))

def describe_asset(model_dir: Path, relative_path: Path) -> dict:
    asset_path = model_dir / relative_path
    asset_hash = hash_file(asset_path)

    if asset_hash is None:
        raise FileNotFoundError(f"Expected model asset at {asset_path}")

    return {
        "filename": relative_path.as_posix(),
        "sha256": asset_hash,
        "bytes": asset_path.stat().st_size
    }

def describe_stage_one(food_signature: dict) -> dict:
    return {
        "input_shape": [1, 3, STAGE_ONE_INPUT_SIZE, STAGE_ONE_INPUT_SIZE],
        "input_layout": "NCHW",
        "input_dtype": "float32 contiguous",
        "channel_order": "RGB",
        "scale": "uint8 / 255",
        "letterbox": {
            "resize": "OpenCV INTER_LINEAR",
            "fill_rgb": [LETTERBOX_FILL, LETTERBOX_FILL, LETTERBOX_FILL],
            "scale": "min(640 / width, 640 / height)",
            "resized_size": "round(original_size * scale)",
            "left_top_pad": "round((640 - resized_size) / 2 - 0.1)",
            "inverse": "subtract pad, divide by scale, clip to original bounds"
        },
        "raw_outputs": [
            {
                "index": 0,
                "shape": food_signature["detection_shape"],
                "dtype": "float32",
                "axes": ["batch", "4 xywh + 1 class + 32 mask coefficients", "anchors"]
            },
            {
                "index": 1,
                "shape": food_signature["prototype_shape"],
                "dtype": "float32",
                "axes": ["batch", "mask channels", "height", "width"]
            }
        ],
        "decode": {
            "head": "one-to-many, no embedded NMS",
            "box_encoding": "center xywh pixels in letterboxed input",
            "class_score": "probability already in [0,1]",
            "score_threshold": SCORE_THRESHOLD,
            "score_comparison": ">",
            "nms": "class agnostic box IoU, score descending then anchor ascending",
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "max_detections": MAX_DETECTIONS,
            "mask": (
                "coefficients @ prototypes, bilinear upsample to the 640x640 input, "
                ">0, then crop to the box at input resolution"
            ),
            "mask_iou_dedupe": MASK_IOU_THRESHOLD,
            "containment_threshold": CONTAINMENT_THRESHOLD,
            "containment_box_iou_guard": CONTAINMENT_BOX_IOU
        }
    }

def describe_stage_two() -> dict:
    return {
        "input_shape": [1, 3, STAGE_TWO_INPUT_SIZE, STAGE_TWO_INPUT_SIZE],
        "input_dtype": "float32 contiguous",
        "crop_rule": "provisional: original-image box plus 20% margin, clipped to image bounds",
        "padding": f"center pad to square with RGB {STAGE_TWO_PADDING}",
        "resize": "OpenCV INTER_LINEAR to 224 x 224",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "output_shape": [1, 7],
        "score_path": "linear head logits then temperature softmax"
    }

def build_manifest(run_dir: Path, model_dir: Path) -> dict:
    run_record = load_json(run_dir / RUN_RECORD_NAME)
    export_report = load_json(run_dir / EXPORT_REPORT_NAME)
    versions = run_record["versions"]
    classes = []

    for class_index, (source_id, name_id) in enumerate(CLASS_ROWS):
        classes.append({"index": class_index, "source_id": source_id, "name_id": name_id})

    return {
        "schema_version": 1,
        "bundle_id": run_record["run_id"],
        "joint_model_version": "spike-0",
        "minimum_app_version": "0.1.0",
        "minimum_decoder_version": "0.1.0",
        "created_at_utc": run_record["started_at_utc"],
        "code_revision": run_record["git_sha"],
        "dataset_manifest_sha256": run_record["manifest_sha256"],
        "split_sha256": None,
        "assets": {
            "stage_one": describe_asset(model_dir, FOOD_PROGRAM_RELATIVE_PATH),
            "stage_two": describe_asset(model_dir, CLASSIFIER_PROGRAM_RELATIVE_PATH)
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": versions["torch"],
            "timm": versions["timm"],
            "ultralytics": versions["ultralytics"],
            "executorch": versions["executorch"],
            "precision": "float32",
            "stage_one_exporter": "Ultralytics ExecuTorch static batch 1 CPU",
            "stage_two_delegate": "XNNPACK static batch 1",
            "target_abis": ["arm64-v8a"]
        },
        "pixels": {
            "exif_orientation": "apply once before geometry, orientations 1-8",
            "color": "RGB uint8",
            "alpha": "convert to RGB",
            "grayscale": "convert to RGB",
            "metadata": "discard after decoding"
        },
        "stage_one": describe_stage_one(export_report["food"]),
        "stage_two": describe_stage_two(),
        "classes": classes,
        "thresholds": {
            "temperature": None,
            "dish_acceptance": None,
            "not_food": None
        }
    }

def check_schema_type(value, expected_type: str) -> bool:
    if expected_type == "null":
        return value is None

    if expected_type == "object":
        return isinstance(value, dict)

    if expected_type == "array":
        return isinstance(value, list)

    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)

    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if expected_type == "string":
        return isinstance(value, str)

    raise ValueError(f"Unsupported schema type {expected_type}")

def check_schema_node(value, rule: dict, definitions: dict, location: str = "manifest") -> None:
    if "$ref" in rule:
        definition_name = rule["$ref"].split("/")[-1]
        check_schema_node(value, definitions[definition_name], definitions, location)

        return

    expected_types = rule.get("type")

    if isinstance(expected_types, str):
        expected_types = [expected_types]

    has_valid_type = True

    if expected_types:
        has_valid_type = any(check_schema_type(value, expected) for expected in expected_types)

    if not has_valid_type:
        raise ValueError(f"Expected {location} type {expected_types}, found {type(value).__name__}")

    if "const" in rule and value != rule["const"]:
        raise ValueError(f"Expected {location} to equal {rule['const']}, found {value}")

    check_schema_bounds(value, rule, location)
    check_schema_children(value, rule, definitions, location)

def check_schema_bounds(value, rule: dict, location: str) -> None:
    if isinstance(value, str) and len(value) < rule.get("minLength", 0):
        raise ValueError(f"Expected longer {location} string")

    if isinstance(value, (int, float)) and value < rule.get("minimum", float("-inf")):
        raise ValueError(f"Expected {location} above minimum")

    if isinstance(value, list) and len(value) < rule.get("minItems", 0):
        raise ValueError(f"Expected more {location} items")

    if isinstance(value, list) and len(value) > rule.get("maxItems", float("inf")):
        raise ValueError(f"Expected fewer {location} items")

def check_schema_children(value, rule: dict, definitions: dict, location: str) -> None:
    if isinstance(value, dict):
        missing_keys = set(rule.get("required", [])) - set(value)

        if missing_keys:
            raise ValueError(f"Expected {location} keys {sorted(missing_keys)}")

        allowed_properties = rule.get("properties", {})

        if rule.get("additionalProperties") is False and set(value) - set(allowed_properties):
            unexpected_keys = sorted(set(value) - set(allowed_properties))
            raise ValueError(f"Unexpected keys at {location}: {unexpected_keys}")

        for property_name, property_rule in allowed_properties.items():
            if property_name in value:
                child_location = f"{location}.{property_name}"
                check_schema_node(value[property_name], property_rule, definitions, child_location)

    if isinstance(value, list) and "items" in rule:
        for item_index, item in enumerate(value):
            check_schema_node(item, rule["items"], definitions, f"{location}[{item_index}]")

def validate_manifest(manifest: dict, schema: dict, model_dir: Path | None = None) -> None:
    check_schema_node(manifest, schema, schema["$defs"])
    class_indices = [class_row["index"] for class_row in manifest["classes"]]
    raw_outputs = manifest["stage_one"]["raw_outputs"]

    if class_indices != list(range(len(class_indices))):
        raise ValueError("Expected stable, consecutive Stage-2 class indices")

    if raw_outputs[0]["shape"] != EXPECTED_DETECTION_SHAPE:
        raise ValueError(f"Expected detection shape {EXPECTED_DETECTION_SHAPE}")

    if raw_outputs[1]["shape"] != EXPECTED_PROTOTYPE_SHAPE:
        raise ValueError(f"Expected prototype shape {EXPECTED_PROTOTYPE_SHAPE}")

    if model_dir is None:
        return

    for asset in manifest["assets"].values():
        relative_path = Path(asset["filename"])

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Expected a relative asset filename, found {relative_path}")

        asset_path = model_dir / relative_path
        actual_hash = hash_file(asset_path)

        if actual_hash is None:
            raise ValueError(f"Asset missing at {asset_path}")

        if actual_hash != asset["sha256"] or asset_path.stat().st_size != asset["bytes"]:
            raise ValueError(f"Asset integrity mismatch at {asset_path}")

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    model_dir = get_data_root_dir() / MODEL_DIR_NAME
    manifest = build_manifest(arguments.run_dir, model_dir)
    schema = load_json(SCHEMA_PATH)
    validate_manifest(manifest, schema, model_dir)
    EXAMPLE_PATH.write_text(json.dumps(manifest, indent=JSON_INDENT) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
