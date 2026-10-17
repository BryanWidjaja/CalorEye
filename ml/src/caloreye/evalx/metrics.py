import math

import numpy as np

WILSON_Z = 1.96

def wilson_interval(successes: int, total: int, z_value: float = WILSON_Z) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0

    proportion = successes / total
    z_squared = z_value ** 2
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    spread_term = proportion * (1.0 - proportion) / total + z_squared / (4 * total ** 2)
    spread = z_value * math.sqrt(spread_term) / denominator

    return max(0.0, center - spread), min(1.0, center + spread)

def confusion_matrix(
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        class_count: int
    ) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)

    for true_label, predicted_label in zip(true_labels, predicted_labels, strict=True):
        matrix[int(true_label), int(predicted_label)] += 1

    return matrix

def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator

def per_class_report(matrix: np.ndarray, class_names: list[str]) -> list[dict]:
    report = []

    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted_total = int(matrix[:, index].sum())
        precision = safe_divide(true_positive, predicted_total)
        recall = safe_divide(true_positive, support)
        f1_score = safe_divide(2 * precision * recall, precision + recall)
        report.append(
            {
                "class": name,
                "support": support,
                "precision": precision,
                "precision_ci": wilson_interval(true_positive, predicted_total),
                "recall": recall,
                "recall_ci": wilson_interval(true_positive, support),
                "f1": f1_score
            }
        )

    return report

def macro_f1(matrix: np.ndarray) -> float:
    scores = []

    for index in range(matrix.shape[0]):
        true_positive = float(matrix[index, index])
        precision = safe_divide(true_positive, float(matrix[:, index].sum()))
        recall = safe_divide(true_positive, float(matrix[index].sum()))
        scores.append(safe_divide(2 * precision * recall, precision + recall))

    return float(np.mean(scores))

def accuracy(matrix: np.ndarray) -> float:
    return safe_divide(float(np.trace(matrix)), float(matrix.sum()))
