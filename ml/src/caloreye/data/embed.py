import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import timm
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from caloreye.data.manifest import (
    ACQUISITION_DIR_NAME,
    LATEST_MANIFEST_NAME,
    WORK_DIR_NAME,
    convert_to_rgb,
)
from caloreye.paths import get_data_root_dir
from caloreye.runs import hash_file, start_run

MODEL_NAME = "vit_small_patch14_reg4_dinov2.lvd142m"
INPUT_SIZE = 224
BATCH_SIZE = 128
WORKER_COUNT = 4
SHARD_SIZE = 2048
HASH_PREFIX_LENGTH = 12
EMBEDDING_DIR_NAME = "embeddings"
SHARD_NAME_FORMAT = "shard-{index:04d}"
IDS_SUFFIX = ".ids.json"
VECTORS_SUFFIX = ".npy"
DRAFT_SIZE = (INPUT_SIZE * 2, INPUT_SIZE * 2)
MIN_FREE_BYTES = 5 * 1024 ** 3
PIXEL_SCALE = 255.0
JSON_INDENT = 2

class ImageDataset(Dataset):
    def __init__(self, image_paths: list[Path]):
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        try:
            with Image.open(self.image_paths[index]) as image:
                image.draft("RGB", DRAFT_SIZE)
                upright = ImageOps.exif_transpose(image)
                rgb_image = convert_to_rgb(upright)
                resized = rgb_image.resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)
                pixel_array = np.asarray(resized, dtype=np.uint8)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            logging.warning("cannot embed %s: %s", self.image_paths[index].name, error)
            pixel_array = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)

        return torch.from_numpy(pixel_array).permute(2, 0, 1).contiguous()

def load_embedding_model() -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=INPUT_SIZE)
    model = model.eval().cuda().half()
    config = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(config["mean"], dtype=torch.float16).view(1, 3, 1, 1).cuda()
    std = torch.tensor(config["std"], dtype=torch.float16).view(1, 3, 1, 1).cuda()

    return model, mean, std

def embed_batch(
        model: torch.nn.Module,
        pixels: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor
    ) -> np.ndarray:
    scaled = pixels.cuda().half() / PIXEL_SCALE
    normalized = (scaled - mean) / std

    with torch.inference_mode():
        features = model(normalized).float()
        features = torch.nn.functional.normalize(features, dim=1)

    return features.half().cpu().numpy()

def embed_shard(
        model: torch.nn.Module,
        mean: torch.Tensor,
        std: torch.Tensor,
        image_paths: list[Path]
    ) -> np.ndarray:
    loader = DataLoader(
        ImageDataset(image_paths),
        batch_size=BATCH_SIZE,
        num_workers=WORKER_COUNT,
        shuffle=False
    )
    parts = []

    for pixels in loader:
        parts.append(embed_batch(model, pixels, mean, std))

    return np.concatenate(parts, axis=0)

def select_unique_images(frame: pl.DataFrame) -> pl.DataFrame:
    decodable = frame.filter(pl.col("decodable"))

    return decodable.unique(subset=["image_id"], keep="first").sort("image_id")

def embedding_dir_for(data_root: Path, manifest_hash: str) -> Path:
    return data_root / WORK_DIR_NAME / EMBEDDING_DIR_NAME / manifest_hash[:HASH_PREFIX_LENGTH]

def load_embeddings(embedding_dir: Path) -> tuple[list[str], np.ndarray]:
    image_ids = []
    parts = []

    for ids_path in sorted(embedding_dir.glob(f"*{IDS_SUFFIX}")):
        stem = ids_path.name[: -len(IDS_SUFFIX)]
        shard_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        image_ids.extend(shard_ids)
        parts.append(np.load(embedding_dir / f"{stem}{VECTORS_SUFFIX}"))

    return image_ids, np.concatenate(parts, axis=0)

def run_embedding(data_root: Path) -> list[str]:
    free_bytes = shutil.disk_usage(data_root).free

    if free_bytes < MIN_FREE_BYTES:
        raise OSError(f"expected at least {MIN_FREE_BYTES} free bytes, found {free_bytes}")

    manifest_path = data_root / WORK_DIR_NAME / LATEST_MANIFEST_NAME
    manifest_hash = hash_file(manifest_path)
    unique = select_unique_images(pl.read_parquet(manifest_path))
    embedding_dir = embedding_dir_for(data_root, manifest_hash)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    acquisition_dir = data_root / ACQUISITION_DIR_NAME
    image_ids = unique["image_id"].to_list()
    relative_paths = unique["path"].to_list()
    model, mean, std = load_embedding_model()
    done_count = 0

    for start in range(0, len(image_ids), SHARD_SIZE):
        name = SHARD_NAME_FORMAT.format(index=start // SHARD_SIZE)
        ids_path = embedding_dir / f"{name}{IDS_SUFFIX}"

        if ids_path.is_file():
            done_count += 1
            continue

        shard_ids = image_ids[start:start + SHARD_SIZE]
        shard_paths = [acquisition_dir / path for path in relative_paths[start:start + SHARD_SIZE]]
        vectors = embed_shard(model, mean, std, shard_paths)
        np.save(embedding_dir / f"{name}{VECTORS_SUFFIX}", vectors)
        ids_path.write_text(json.dumps(shard_ids, indent=JSON_INDENT), encoding="utf-8")
        logging.warning("embedded %d of %d", start + len(shard_ids), len(image_ids))

    return [f"embeddings {len(image_ids)} images, {done_count} shards reused, dir {embedding_dir}"]

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye data embed")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    arguments = parser.parse_args(argv)
    config = {
        "model": MODEL_NAME,
        "input_size": INPUT_SIZE,
        "batch_size": BATCH_SIZE,
        "shard_size": SHARD_SIZE
    }
    start_run("data-embed", config)

    return run_embedding(arguments.data_root)
