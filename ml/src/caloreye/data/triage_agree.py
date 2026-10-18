import argparse
import hashlib
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import torch

from caloreye.paths import get_data_root_dir
from caloreye.runs import start_run

SEED = 42
FOLD_COUNT = 5
TRAIN_EPOCHS = 10
TRAIN_BATCH = 4096
TRAIN_RATE = 0.05
GPU_CHUNK = 1024
NEIGHBOR_COUNT = 5
CLUSTER_COSINE = 0.985
CENTROID_FRACTION = 0.7
AGREE_VERSION = "1"
EPSILON = 1e-8
TRUSTED_ROUTES = ("food101", "uec256")

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(lengths, EPSILON)

def assign_folds(candidate_ids: list[str]) -> np.ndarray:
    folds = []

    for candidate_id in candidate_ids:
        digest = hashlib.sha256(candidate_id.encode()).digest()
        folds.append(int.from_bytes(digest[:4], "big") % FOLD_COUNT)

    return np.asarray(folds, dtype=np.int8)

def refine_prototype(vectors: np.ndarray, center: np.ndarray) -> np.ndarray:
    similarities = vectors @ center
    cutoff = np.quantile(similarities, 1 - CENTROID_FRACTION)
    refined = np.median(vectors[similarities >= cutoff], axis=0)
    refined /= np.linalg.norm(refined) + EPSILON

    return refined

def robust_prototype(vectors: np.ndarray) -> np.ndarray:
    center = np.median(vectors, axis=0)
    center /= np.linalg.norm(center) + EPSILON

    center = refine_prototype(vectors, center)

    return refine_prototype(vectors, center)

def build_prototypes(vectors: np.ndarray, labels: np.ndarray, routes: list[str], class_count: int) -> tuple[np.ndarray, np.ndarray]:
    trusted = np.isin(routes, TRUSTED_ROUTES)
    prototypes = np.zeros((class_count, vectors.shape[1]), dtype=np.float32)
    trusted_counts = np.zeros(class_count, dtype=np.int32)

    for class_index in range(class_count):
        members = labels == class_index
        trusted_members = members & trusted
        trusted_counts[class_index] = trusted_members.sum()

        if trusted_counts[class_index]:
            prototypes[class_index] = robust_prototype(vectors[trusted_members])
        elif members.any():
            prototypes[class_index] = robust_prototype(vectors[members])

    return prototypes, trusted_counts

def fit_out_of_fold(vectors: np.ndarray, labels: np.ndarray, folds: np.ndarray, class_count: int) -> np.ndarray:
    torch.manual_seed(SEED)
    features = torch.from_numpy(vectors).to("cuda", dtype=torch.float16)
    targets = torch.from_numpy(labels.astype(np.int64)).to("cuda")
    probabilities = np.zeros(len(labels), dtype=np.float32)

    for fold in range(FOLD_COUNT):
        train_indices = np.flatnonzero(folds != fold)
        holdout_indices = np.flatnonzero(folds == fold)
        model = torch.nn.Linear(vectors.shape[1], class_count).to("cuda", dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_RATE)

        for epoch in range(TRAIN_EPOCHS):
            shuffled = torch.from_numpy(np.random.default_rng(SEED + fold * TRAIN_EPOCHS + epoch).permutation(train_indices))

            for offset in range(0, len(shuffled), TRAIN_BATCH):
                indices = shuffled[offset:offset + TRAIN_BATCH].to("cuda")
                logits = model(features[indices].float())
                loss = torch.nn.functional.cross_entropy(logits, targets[indices])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.inference_mode():
            for offset in range(0, len(holdout_indices), TRAIN_BATCH):
                positions = holdout_indices[offset:offset + TRAIN_BATCH]
                indices = torch.from_numpy(positions).to("cuda")
                logits = model(features[indices].float())
                scores = torch.softmax(logits, dim=1)
                selected = scores.gather(1, targets[indices, None]).squeeze(1)
                probabilities[positions] = selected.cpu().numpy()

        logging.info("finished out-of-fold split %s/%s", fold + 1, FOLD_COUNT)

    return probabilities

def given_label_suspects(probabilities: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    sums = np.bincount(labels, weights=probabilities, minlength=class_count)
    counts = np.bincount(labels, minlength=class_count)
    means = sums / np.maximum(counts, 1)
    return probabilities < means[labels]

def trusted_neighbor_agreement(vectors: np.ndarray, labels: np.ndarray, routes: list[str], trusted_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trusted_mask = np.isin(routes, TRUSTED_ROUTES)
    trusted_indices = np.flatnonzero(trusted_mask)
    agreements = np.full(len(labels), -1, dtype=np.int8)
    similarities = np.zeros(len(labels), dtype=np.float32)

    if len(trusted_indices) == 0:
        return agreements, similarities

    features = torch.from_numpy(vectors).to("cuda", dtype=torch.float16)
    reference = features[trusted_indices]
    reference_labels = labels[trusted_indices]

    for offset in range(0, len(labels), GPU_CHUNK):
        scores = features[offset:offset + GPU_CHUNK] @ reference.T
        best_scores, best_positions = torch.topk(scores, min(NEIGHBOR_COUNT + 1, len(trusted_indices)), dim=1)

        for local_index, (values, positions) in enumerate(zip(best_scores.cpu().numpy(), best_positions.cpu().numpy(), strict=True)):
            image_index = offset + local_index

            if trusted_counts[labels[image_index]] == 0:
                continue

            neighbor_labels = []
            neighbor_scores = []

            for value, position in zip(values, positions, strict=True):
                if trusted_indices[position] == image_index:
                    continue

                neighbor_labels.append(reference_labels[position])
                neighbor_scores.append(value)

                if len(neighbor_labels) == NEIGHBOR_COUNT:
                    break

            if neighbor_labels:
                agreements[image_index] = int(Counter(neighbor_labels).most_common(1)[0][0] == labels[image_index])
                similarities[image_index] = float(max(neighbor_scores))

    return agreements, similarities

def find_root(parents: list[int], index: int) -> int:
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]

    return index

def cluster_embeddings(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = torch.from_numpy(vectors).to("cuda", dtype=torch.float16)
    parents = list(range(len(vectors)))

    for offset in range(0, len(vectors), GPU_CHUNK):
        similarities = features[offset:offset + GPU_CHUNK] @ features.T
        best_scores, best_indices = torch.topk(similarities, NEIGHBOR_COUNT + 1, dim=1)

        for local_index, (values, indices) in enumerate(zip(best_scores.cpu().numpy(), best_indices.cpu().numpy(), strict=True)):
            image_index = offset + local_index

            for value, neighbor_index in zip(values, indices, strict=True):
                if neighbor_index == image_index or value < CLUSTER_COSINE:
                    continue

                first_root = find_root(parents, image_index)
                second_root = find_root(parents, int(neighbor_index))
                parents[second_root] = first_root

    clusters = np.asarray([find_root(parents, index) for index in range(len(parents))], dtype=np.int32)
    sizes = np.bincount(clusters, minlength=len(clusters))
    return clusters, sizes[clusters]

def run_agreement(data_root: Path) -> tuple[Path, int]:
    score_path = data_root / "acquisition" / "clean" / "score.parquet"

    if not score_path.is_file():
        raise FileNotFoundError(f"missing model scores: {score_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("label agreement requires a CUDA GPU")

    logging.basicConfig(filename=data_root / "triage_agree.log", level=logging.INFO, format="%(message)s", force=True)
    start_run("data-triage-agree", {"version": AGREE_VERSION, "folds": FOLD_COUNT, "cluster_cosine": CLUSTER_COSINE})
    frame = pl.read_parquet(score_path, columns=["candidate_id", "source_id", "discovery_route", "embedding"])
    vectors = normalize_vectors(np.stack([np.frombuffer(value, dtype=np.float16).astype(np.float32) for value in frame["embedding"]]))
    class_ids = sorted(frame["source_id"].unique().to_list())
    class_positions = {source_id: position for position, source_id in enumerate(class_ids)}
    labels = np.asarray([class_positions[source_id] for source_id in frame["source_id"]], dtype=np.int32)
    routes = frame["discovery_route"].to_list()
    folds = assign_folds(frame["candidate_id"].to_list())
    prototypes, trusted_counts = build_prototypes(vectors, labels, routes, len(class_ids))
    prototype_scores = vectors @ prototypes.T
    given_similarity = prototype_scores[np.arange(len(labels)), labels]
    prototype_scores[np.arange(len(labels)), labels] = -1
    prototype_margin = given_similarity - prototype_scores.max(axis=1)
    probabilities = fit_out_of_fold(vectors, labels, folds, len(class_ids))
    suspects = given_label_suspects(probabilities, labels, len(class_ids))
    agreements, neighbor_similarity = trusted_neighbor_agreement(vectors, labels, routes, trusted_counts)
    clusters, cluster_sizes = cluster_embeddings(vectors)
    output_path = score_path.parent / "agree.parquet"
    temporary_path = output_path.with_suffix(".tmp")
    pl.DataFrame({
        "candidate_id": frame["candidate_id"],
        "source_id": frame["source_id"],
        "discovery_route": frame["discovery_route"],
        "given_label_probability": probabilities,
        "confident_learning_suspect": suspects,
        "prototype_similarity": given_similarity,
        "prototype_margin": prototype_margin,
        "trusted_prototype_count": trusted_counts[labels],
        "trusted_knn_agreement": agreements,
        "trusted_knn_similarity": neighbor_similarity,
        "embedding_cluster": clusters,
        "embedding_cluster_size": cluster_sizes,
        "version": [AGREE_VERSION] * len(labels)
    }).write_parquet(temporary_path)
    temporary_path.replace(output_path)
    return output_path, len(labels)

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye data triage agree")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    arguments = parser.parse_args(argv)
    output_path, count = run_agreement(arguments.data_root)
    return [f"checked agreement for {count} images", f"agree {output_path}"]
