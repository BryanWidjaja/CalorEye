from dataclasses import dataclass

import numpy as np

HASH_BIT_COUNT = 64
CHUNK_COUNT = 5
HAMMING_THRESHOLD = 4
CHUNK_BIT_WIDTHS = (13, 13, 13, 13, 12)

@dataclass(frozen=True)
class ClusterResult:
    cluster_ids: np.ndarray
    near_pair_count: int


def hamming_distances(query_hash: np.uint64, other_hashes: np.ndarray) -> np.ndarray:
    differing_bits = np.bitwise_xor(other_hashes, query_hash)

    return np.bitwise_count(differing_bits)


def extract_chunk(hashes: np.ndarray, chunk_index: int) -> np.ndarray:
    shift_bits = sum(CHUNK_BIT_WIDTHS[:chunk_index])
    mask = np.uint64((1 << CHUNK_BIT_WIDTHS[chunk_index]) - 1)
    shifted = np.right_shift(hashes, np.uint64(shift_bits))

    return np.bitwise_and(shifted, mask)


def find_root(parents: np.ndarray, node: int) -> int:
    root = node

    while parents[root] != root:
        root = int(parents[root])

    while parents[node] != root:
        next_node = int(parents[node])
        parents[node] = root
        node = next_node

    return root


def union_nodes(parents: np.ndarray, first: int, second: int) -> bool:
    first_root = find_root(parents, first)
    second_root = find_root(parents, second)

    if first_root == second_root:
        return False

    parents[max(first_root, second_root)] = min(first_root, second_root)

    return True


def find_near_pairs(
        hashes: np.ndarray,
        max_distance: int = HAMMING_THRESHOLD
    ) -> list[tuple[int, int]]:
    pairs = set()

    for chunk_index in range(CHUNK_COUNT):
        chunk_values = extract_chunk(hashes, chunk_index)
        order = np.argsort(chunk_values, kind="stable")
        sorted_values = chunk_values[order]
        boundaries = np.flatnonzero(np.diff(sorted_values)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(order)]))

        for start, end in zip(starts, ends, strict=True):
            if end - start < 2:
                continue

            members = order[start:end]
            collect_bucket_pairs(hashes, members, max_distance, pairs)

    return sorted(pairs)


def collect_bucket_pairs(
        hashes: np.ndarray,
        members: np.ndarray,
        max_distance: int,
        pairs: set
    ) -> None:
    member_hashes = hashes[members]

    for position in range(len(members) - 1):
        distances = hamming_distances(member_hashes[position], member_hashes[position + 1:])
        close_positions = np.flatnonzero(distances <= max_distance)

        for offset in close_positions:
            first = int(members[position])
            second = int(members[position + 1 + offset])
            pairs.add((min(first, second), max(first, second)))


def cluster_by_hash(
        image_ids: list[str],
        phashes: np.ndarray,
        max_distance: int = HAMMING_THRESHOLD
    ) -> ClusterResult:
    unique_ids = sorted(set(image_ids))
    first_index = {}

    for position, image_id in enumerate(image_ids):
        if image_id not in first_index:
            first_index[image_id] = position

    unique_hashes = np.zeros(len(unique_ids), dtype=np.uint64)

    for position, image_id in enumerate(unique_ids):
        unique_hashes[position] = phashes[first_index[image_id]]

    pairs = find_near_pairs(unique_hashes, max_distance)
    parents = np.arange(len(unique_ids))

    for first, second in pairs:
        union_nodes(parents, first, second)

    unique_position = {image_id: position for position, image_id in enumerate(unique_ids)}
    cluster_ids = np.zeros(len(image_ids), dtype=np.int64)

    for position, image_id in enumerate(image_ids):
        cluster_ids[position] = find_root(parents, unique_position[image_id])

    return ClusterResult(cluster_ids=cluster_ids, near_pair_count=len(pairs))
