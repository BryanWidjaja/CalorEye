from dataclasses import dataclass

import numpy as np

COVERAGE_LEVELS = (0.95, 0.9, 0.75)
FALSE_ACCEPT_KEY = "false_accept_at_{coverage}"

@dataclass(frozen=True)
class VimModel:
    origin: np.ndarray
    null_space: np.ndarray
    alpha: float
    dimension: int

def compute_log_sum_exp(logits: np.ndarray) -> np.ndarray:
    maxima = logits.max(axis=1, keepdims=True)
    shifted = np.exp(logits - maxima).sum(axis=1)

    return maxima[:, 0] + np.log(shifted)

def compute_max_logit(logits: np.ndarray) -> np.ndarray:
    return logits.max(axis=1)

def compute_max_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    shifted = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probabilities = shifted / shifted.sum(axis=1, keepdims=True)

    return probabilities.max(axis=1)

def compute_residual_norm(
        features: np.ndarray,
        origin: np.ndarray,
        null_space: np.ndarray
    ) -> np.ndarray:
    projected = (features - origin) @ null_space

    return np.linalg.norm(projected, axis=1)

def fit_vim(
        train_features: np.ndarray,
        train_logits: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
        dimension: int
    ) -> VimModel:
    origin = -np.linalg.pinv(weights) @ bias
    centered = train_features - origin
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    null_space = eigenvectors[:, order[dimension:]]
    residual_norms = compute_residual_norm(train_features, origin, null_space)
    alpha = float(train_logits.max(axis=1).mean() / residual_norms.mean())

    return VimModel(origin, null_space, alpha, dimension)

def score_vim(model: VimModel, features: np.ndarray, logits: np.ndarray) -> np.ndarray:
    residual_norms = compute_residual_norm(features, model.origin, model.null_space)

    return compute_log_sum_exp(logits) - model.alpha * residual_norms

def calculate_auroc(known_scores: np.ndarray, unseen_scores: np.ndarray) -> float:
    pooled = np.sort(np.concatenate([known_scores, unseen_scores]))
    lower = np.searchsorted(pooled, known_scores, side="left")
    upper = np.searchsorted(pooled, known_scores, side="right")
    known_ranks = (lower + upper + 1) / 2
    rank_sum = known_ranks.sum()
    known_count = len(known_scores)
    unseen_count = len(unseen_scores)
    statistic = rank_sum - known_count * (known_count + 1) / 2

    return float(statistic / (known_count * unseen_count))

def find_threshold_for_coverage(known_scores: np.ndarray, coverage: float) -> float:
    return float(np.quantile(known_scores, 1.0 - coverage))

def calculate_false_accept_rate(unseen_scores: np.ndarray, threshold: float) -> float:
    return float((unseen_scores >= threshold).mean())

def summarize_scores(known_scores: np.ndarray, unseen_scores: np.ndarray) -> dict:
    summary = {"auroc": calculate_auroc(known_scores, unseen_scores)}

    for coverage in COVERAGE_LEVELS:
        threshold = find_threshold_for_coverage(known_scores, coverage)
        key = FALSE_ACCEPT_KEY.format(coverage=coverage)
        summary[key] = calculate_false_accept_rate(unseen_scores, threshold)

    return summary
