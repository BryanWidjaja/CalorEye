import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NUTRITION_PATH = Path(__file__).resolve().parents[3] / "contracts" / "nutrition_v0.csv"
PROVISIONAL_MARK = "[provisional]"
GRAMS_PER_HUNDRED = 100.0

@dataclass(frozen=True)
class Nutrition:
    class_name: str
    kcal_per_100g: float
    default_portion_g: float
    status: str

@dataclass(frozen=True)
class PlateItem:
    class_name: str
    grams: float

def load_nutrition(path: Path = NUTRITION_PATH) -> dict[str, Nutrition]:
    table = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            table[record["class_name"]] = Nutrition(
                class_name=record["class_name"],
                kcal_per_100g=float(record["kcal_per_100g"]),
                default_portion_g=float(record["default_portion_g"]),
                status=record["status"]
            )

    return table

def item_kcal(table: dict[str, Nutrition], class_name: str, grams: float) -> float:
    return table[class_name].kcal_per_100g * grams / GRAMS_PER_HUNDRED

def truth_kcal(table: dict[str, Nutrition], truth_items: list[PlateItem]) -> float:
    return sum(item_kcal(table, item.class_name, item.grams) for item in truth_items)

def default_portion_kcal(table: dict[str, Nutrition], predicted_classes: list[str]) -> float:
    total = 0.0

    for class_name in predicted_classes:
        total += item_kcal(table, class_name, table[class_name].default_portion_g)

    return total

def oracle_urt_kcal(
        table: dict[str, Nutrition],
        truth_items: list[PlateItem],
        predicted_classes: list[str],
        matches: list[tuple[int, int]]
    ) -> float:
    total = 0.0
    matched_predictions = set()

    for prediction_index, truth_index in matches:
        matched_predictions.add(prediction_index)
        predicted_class = predicted_classes[prediction_index]
        total += item_kcal(table, predicted_class, truth_items[truth_index].grams)

    for prediction_index, class_name in enumerate(predicted_classes):
        if prediction_index not in matched_predictions:
            total += item_kcal(table, class_name, table[class_name].default_portion_g)

    return total

def plate_errors(
        table: dict[str, Nutrition],
        truth_items: list[PlateItem],
        predicted_classes: list[str],
        matches: list[tuple[int, int]]
    ) -> dict:
    truth_total = truth_kcal(table, truth_items)
    default_total = default_portion_kcal(table, predicted_classes)
    oracle_total = oracle_urt_kcal(table, truth_items, predicted_classes, matches)

    return {
        "truth_kcal": truth_total,
        "default_portion_kcal": default_total,
        "oracle_urt_kcal": oracle_total,
        "default_portion_error": default_total - truth_total,
        "oracle_urt_error": oracle_total - truth_total
    }

def summarize_plates(plates: list[dict]) -> dict:
    summary = {"plates": len(plates), "status": PROVISIONAL_MARK}

    for mode in ("default_portion", "oracle_urt"):
        errors = np.array([plate[f"{mode}_error"] for plate in plates])
        truths = np.array([plate["truth_kcal"] for plate in plates])
        percentage_errors = np.abs(errors) / np.maximum(truths, 1.0)
        summary[mode] = {
            "mean_absolute_error_kcal": float(np.abs(errors).mean()),
            "mean_signed_error_kcal": float(errors.mean()),
            "mean_absolute_percentage_error": float(percentage_errors.mean())
        }

    return summary
