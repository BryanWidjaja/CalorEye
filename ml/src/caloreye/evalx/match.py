from dataclasses import dataclass

import numpy as np

MATCH_IOU_THRESHOLD = 0.5
IOA_THRESHOLD = 0.5
MIN_MULTIPLE = 2

@dataclass
class MatchResult:
    matches: list[tuple[int, int, float]]
    unmatched_predictions: list[int]
    unmatched_truths: list[int]
    merge_count: int
    split_count: int

def flatten_masks(masks: np.ndarray) -> np.ndarray:
    return masks.reshape(masks.shape[0], -1).astype(np.float32)

def intersection_matrix(prediction_masks: np.ndarray, truth_masks: np.ndarray) -> np.ndarray:
    return flatten_masks(prediction_masks) @ flatten_masks(truth_masks).T

def mask_iou_matrix(prediction_masks: np.ndarray, truth_masks: np.ndarray) -> np.ndarray:
    intersections = intersection_matrix(prediction_masks, truth_masks)
    prediction_areas = flatten_masks(prediction_masks).sum(axis=1)
    truth_areas = flatten_masks(truth_masks).sum(axis=1)
    unions = prediction_areas[:, None] + truth_areas[None, :] - intersections

    return intersections / np.maximum(unions, 1.0)

def greedy_match(
        iou_matrix: np.ndarray,
        threshold: float = MATCH_IOU_THRESHOLD
    ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates = np.argwhere(iou_matrix >= threshold)
    scores = iou_matrix[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(-scores)
    used_predictions = set()
    used_truths = set()
    matches = []

    for position in order:
        prediction_index = int(candidates[position, 0])
        truth_index = int(candidates[position, 1])

        if prediction_index in used_predictions or truth_index in used_truths:
            continue

        used_predictions.add(prediction_index)
        used_truths.add(truth_index)
        score = float(iou_matrix[prediction_index, truth_index])
        matches.append((prediction_index, truth_index, score))

    unmatched_predictions = sorted(set(range(iou_matrix.shape[0])) - used_predictions)
    unmatched_truths = sorted(set(range(iou_matrix.shape[1])) - used_truths)

    return matches, unmatched_predictions, unmatched_truths

def count_merges_and_splits(
        prediction_masks: np.ndarray,
        truth_masks: np.ndarray,
        threshold: float = IOA_THRESHOLD
    ) -> tuple[int, int]:
    intersections = intersection_matrix(prediction_masks, truth_masks)
    prediction_areas = np.maximum(flatten_masks(prediction_masks).sum(axis=1), 1.0)
    truth_areas = np.maximum(flatten_masks(truth_masks).sum(axis=1), 1.0)
    truth_covered = intersections / truth_areas[None, :] >= threshold
    prediction_inside = intersections / prediction_areas[:, None] >= threshold
    merge_count = int((truth_covered.sum(axis=1) >= MIN_MULTIPLE).sum())
    split_count = int((prediction_inside.sum(axis=0) >= MIN_MULTIPLE).sum())

    return merge_count, split_count

def match_instances(prediction_masks: np.ndarray, truth_masks: np.ndarray) -> MatchResult:
    if len(prediction_masks) == 0 or len(truth_masks) == 0:
        return MatchResult(
            matches=[],
            unmatched_predictions=list(range(len(prediction_masks))),
            unmatched_truths=list(range(len(truth_masks))),
            merge_count=0,
            split_count=0
        )

    iou_matrix = mask_iou_matrix(prediction_masks, truth_masks)
    matches, unmatched_predictions, unmatched_truths = greedy_match(iou_matrix)
    merge_count, split_count = count_merges_and_splits(prediction_masks, truth_masks)

    return MatchResult(matches, unmatched_predictions, unmatched_truths, merge_count, split_count)

def summarize_matches(results: list[MatchResult]) -> dict:
    matched = sum(len(result.matches) for result in results)
    predicted = matched + sum(len(result.unmatched_predictions) for result in results)
    truth = matched + sum(len(result.unmatched_truths) for result in results)

    return {
        "images": len(results),
        "matched": matched,
        "predicted": predicted,
        "truth": truth,
        "recall": matched / max(truth, 1),
        "precision": matched / max(predicted, 1),
        "merges": sum(result.merge_count for result in results),
        "splits": sum(result.split_count for result in results)
    }
