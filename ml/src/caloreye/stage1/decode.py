from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps

INPUT_SIZE = 640
LETTERBOX_FILL = 114
PIXEL_SCALE = 255
BOX_CHANNEL_COUNT = 4
MASK_CHANNEL_COUNT = 32
DETECTION_TENSOR_RANK = 3
PROTOTYPE_TENSOR_RANK = 4
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_NMS_IOU_THRESHOLD = 0.7
DEFAULT_MAX_DETECTIONS = 300
DEFAULT_MASK_IOU_THRESHOLD = 0.8
DEFAULT_CONTAINMENT_THRESHOLD = 0.9
DEFAULT_BOX_IOU_GUARD = 0.5
ROUNDING_BIAS = 0.1

@dataclass(frozen=True)
class LetterboxParams:
    original_width_px: int
    original_height_px: int
    input_width_px: int
    input_height_px: int
    resized_width_px: int
    resized_height_px: int
    pad_left_px: int
    pad_top_px: int
    scale: float

@dataclass(frozen=True)
class DecodeSettings:
    class_count: int = 1
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD
    max_detections: int = DEFAULT_MAX_DETECTIONS
    remove_contained: bool = True

@dataclass(frozen=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    anchor_index: int
    mask: np.ndarray

@dataclass(frozen=True)
class Candidate:
    box_xyxy: np.ndarray
    score: float
    class_id: int
    anchor_index: int

def load_image_rgb(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as source:
        image_rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))

    return image_rgb

def calculate_letterbox(original_width_px: int, original_height_px: int) -> LetterboxParams:
    scale = min(INPUT_SIZE / original_width_px, INPUT_SIZE / original_height_px)
    resized_width_px = round(original_width_px * scale)
    resized_height_px = round(original_height_px * scale)
    horizontal_padding = (INPUT_SIZE - resized_width_px) / 2
    vertical_padding = (INPUT_SIZE - resized_height_px) / 2

    return LetterboxParams(
        original_width_px=original_width_px,
        original_height_px=original_height_px,
        input_width_px=INPUT_SIZE,
        input_height_px=INPUT_SIZE,
        resized_width_px=resized_width_px,
        resized_height_px=resized_height_px,
        pad_left_px=round(horizontal_padding - ROUNDING_BIAS),
        pad_top_px=round(vertical_padding - ROUNDING_BIAS),
        scale=scale
    )

def prepare_stage_one_input(image_rgb: np.ndarray) -> tuple[torch.Tensor, LetterboxParams]:
    original_height_px, original_width_px = image_rgb.shape[:2]
    params = calculate_letterbox(original_width_px, original_height_px)
    resized = cv2.resize(
        image_rgb,
        (params.resized_width_px, params.resized_height_px),
        interpolation=cv2.INTER_LINEAR
    )
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), LETTERBOX_FILL, dtype=np.uint8)
    bottom_px = params.pad_top_px + params.resized_height_px
    right_px = params.pad_left_px + params.resized_width_px
    canvas[params.pad_top_px:bottom_px, params.pad_left_px:right_px] = resized

    channels_first = np.ascontiguousarray(canvas.transpose(2, 0, 1))
    input_tensor = torch.from_numpy(channels_first).float().div(PIXEL_SCALE).unsqueeze(0)

    return input_tensor.contiguous(), params

def convert_xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    boxes_xyxy = boxes_xywh.copy()
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    return boxes_xyxy

def calculate_box_iou(first_box: np.ndarray, second_box: np.ndarray) -> float:
    overlap_left = max(first_box[0], second_box[0])
    overlap_top = max(first_box[1], second_box[1])
    overlap_right = min(first_box[2], second_box[2])
    overlap_bottom = min(first_box[3], second_box[3])
    overlap_area = max(overlap_right - overlap_left, 0) * max(overlap_bottom - overlap_top, 0)
    first_area = max(first_box[2] - first_box[0], 0) * max(first_box[3] - first_box[1], 0)
    second_area = max(second_box[2] - second_box[0], 0) * max(second_box[3] - second_box[1], 0)
    union_area = first_area + second_area - overlap_area

    if union_area <= 0:
        return 0.0

    return float(overlap_area / union_area)

def select_nms_indices(
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        max_detections: int
    ) -> list[int]:
    anchor_indices = np.arange(len(scores))
    ranked_indices = np.lexsort((anchor_indices, -scores))
    kept_indices = []

    for candidate_index in ranked_indices:
        candidate_box = boxes_xyxy[candidate_index]
        has_overlap = False

        for kept_index in kept_indices:
            if calculate_box_iou(candidate_box, boxes_xyxy[kept_index]) > threshold:
                has_overlap = True
                break

        if not has_overlap:
            kept_indices.append(int(candidate_index))

        if len(kept_indices) >= max_detections:
            break

    return kept_indices

def restore_original_box(box_xyxy: np.ndarray, params: LetterboxParams) -> tuple:
    left = np.clip((box_xyxy[0] - params.pad_left_px) / params.scale, 0, params.original_width_px)
    top = np.clip((box_xyxy[1] - params.pad_top_px) / params.scale, 0, params.original_height_px)
    right = np.clip((box_xyxy[2] - params.pad_left_px) / params.scale, 0, params.original_width_px)
    bottom = np.clip((box_xyxy[3] - params.pad_top_px) / params.scale, 0, params.original_height_px)

    return float(left), float(top), float(right), float(bottom)

def restore_original_mask(mask: np.ndarray, params: LetterboxParams) -> np.ndarray:
    bottom_px = params.pad_top_px + params.resized_height_px
    right_px = params.pad_left_px + params.resized_width_px
    unpadded_mask = mask[params.pad_top_px:bottom_px, params.pad_left_px:right_px]

    return cv2.resize(
        unpadded_mask,
        (params.original_width_px, params.original_height_px),
        interpolation=cv2.INTER_NEAREST
    )

def compute_mask_logits(coefficients: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    mask_channels, mask_height, mask_width = prototypes.shape
    mask_logits = coefficients @ prototypes.reshape(mask_channels, -1).float()

    return mask_logits.reshape(mask_height, mask_width)

def crop_binary_mask(binary_mask: torch.Tensor, box_xyxy: np.ndarray) -> torch.Tensor:
    mask_height, mask_width = binary_mask.shape
    horizontal_pixels = torch.arange(mask_width)[None, :]
    vertical_pixels = torch.arange(mask_height)[:, None]
    inside_box = (
        (horizontal_pixels >= box_xyxy[0])
        & (horizontal_pixels < box_xyxy[2])
        & (vertical_pixels >= box_xyxy[1])
        & (vertical_pixels < box_xyxy[3])
    )

    return binary_mask * inside_box

def decode_mask(
        coefficients: torch.Tensor,
        prototypes: torch.Tensor,
        box_xyxy: np.ndarray,
        params: LetterboxParams
    ) -> np.ndarray:
    mask_logits = compute_mask_logits(coefficients, prototypes)
    upsampled_logits = functional.interpolate(
        mask_logits[None, None],
        size=(params.input_height_px, params.input_width_px),
        mode="bilinear"
    )[0, 0]
    binary_mask = upsampled_logits.gt(0).byte()
    cropped_mask = crop_binary_mask(binary_mask, box_xyxy)

    return restore_original_mask(cropped_mask.numpy(), params)

def is_duplicate(candidate: Detection, kept: Detection) -> bool:
    overlap_pixels = np.logical_and(candidate.mask, kept.mask).sum()
    candidate_pixels = np.count_nonzero(candidate.mask)
    kept_pixels = np.count_nonzero(kept.mask)
    smaller_area = min(candidate_pixels, kept_pixels)
    union_pixels = candidate_pixels + kept_pixels - overlap_pixels

    if smaller_area == 0 or union_pixels == 0:
        return False

    mask_iou = overlap_pixels / union_pixels
    containment = overlap_pixels / smaller_area
    box_iou = calculate_box_iou(np.asarray(candidate.box_xyxy), np.asarray(kept.box_xyxy))

    return mask_iou >= DEFAULT_MASK_IOU_THRESHOLD or (
        containment >= DEFAULT_CONTAINMENT_THRESHOLD and box_iou >= DEFAULT_BOX_IOU_GUARD
    )

def remove_duplicate_detections(detections: list[Detection]) -> list[Detection]:
    kept_detections = []

    for candidate in detections:
        has_duplicate = False

        for kept in kept_detections:
            if is_duplicate(candidate, kept):
                has_duplicate = True
                break

        if not has_duplicate:
            kept_detections.append(candidate)

    return kept_detections

def validate_raw_signature(
        raw_detections: torch.Tensor,
        raw_prototypes: torch.Tensor,
        class_count: int
    ) -> None:
    expected_channels = BOX_CHANNEL_COUNT + class_count + MASK_CHANNEL_COUNT

    has_detection_rank = raw_detections.ndim == DETECTION_TENSOR_RANK
    has_detection_channels = raw_detections.shape[:2] == (1, expected_channels)

    if not has_detection_rank or not has_detection_channels:
        raise ValueError(f"Expected detection shape (1, {expected_channels}, anchors)")

    has_prototype_rank = raw_prototypes.ndim == PROTOTYPE_TENSOR_RANK
    has_prototype_channels = raw_prototypes.shape[:2] == (1, MASK_CHANNEL_COUNT)

    if not has_prototype_rank or not has_prototype_channels:
        raise ValueError(f"Expected prototype shape (1, {MASK_CHANNEL_COUNT}, height, width)")

def select_candidates(raw_detections: torch.Tensor, settings: DecodeSettings) -> tuple:
    scores_by_class = raw_detections[0, BOX_CHANNEL_COUNT:BOX_CHANNEL_COUNT + settings.class_count]
    best_scores, best_classes = scores_by_class.max(dim=0)
    eligible_anchors = torch.where(best_scores > settings.score_threshold)[0]

    if len(eligible_anchors) == 0:
        return eligible_anchors, np.empty(0), best_classes, np.empty((0, BOX_CHANNEL_COUNT))

    scores = best_scores[eligible_anchors].detach().cpu().numpy()
    boxes_xywh = raw_detections[0, :BOX_CHANNEL_COUNT, eligible_anchors].T.detach().cpu().numpy()
    boxes_xyxy = convert_xywh_to_xyxy(boxes_xywh)

    return eligible_anchors, scores, best_classes, boxes_xyxy

def decode_candidate(
        raw_detections: torch.Tensor,
        raw_prototypes: torch.Tensor,
        params: LetterboxParams,
        settings: DecodeSettings,
        candidate: Candidate
    ) -> Detection | None:
    coefficient_start = BOX_CHANNEL_COUNT + settings.class_count
    coefficients = raw_detections[0, coefficient_start:, candidate.anchor_index]
    mask = decode_mask(coefficients, raw_prototypes[0], candidate.box_xyxy, params)

    if not mask.any():
        return None

    return Detection(
        box_xyxy=restore_original_box(candidate.box_xyxy, params),
        score=candidate.score,
        class_id=candidate.class_id,
        anchor_index=candidate.anchor_index,
        mask=mask
    )

def decode_stage_one(
        raw_detections: torch.Tensor,
        raw_prototypes: torch.Tensor,
        params: LetterboxParams,
        settings: DecodeSettings
    ) -> list[Detection]:
    validate_raw_signature(raw_detections, raw_prototypes, settings.class_count)
    eligible_anchors, scores, best_classes, boxes_xyxy = select_candidates(raw_detections, settings)

    if len(eligible_anchors) == 0:
        return []

    kept_indices = select_nms_indices(
        boxes_xyxy,
        scores,
        settings.nms_iou_threshold,
        settings.max_detections
    )
    detections = []

    for kept_index in kept_indices:
        anchor_index = int(eligible_anchors[kept_index])
        candidate_data = Candidate(
            box_xyxy=boxes_xyxy[kept_index],
            score=float(scores[kept_index]),
            class_id=int(best_classes[anchor_index]),
            anchor_index=anchor_index
        )
        candidate = decode_candidate(
            raw_detections,
            raw_prototypes,
            params,
            settings,
            candidate_data
        )

        if candidate is None:
            continue

        detections.append(candidate)

    if settings.remove_contained:
        return remove_duplicate_detections(detections)

    return detections
