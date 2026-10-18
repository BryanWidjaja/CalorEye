import argparse
import csv
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

from caloreye.data.manifest import convert_to_rgb
from caloreye.paths import REPO_ROOT_DIR, get_data_root_dir
from caloreye.runs import start_run
from caloreye.taxonomy import CLASS_MAP_PATH, read_class_map

MODEL_ID = "google/siglip2-base-patch16-384"
MODEL_REVISION = "f775b65a79762255128c981547af89addcfe0f88"
SCORE_VERSION = "1"
IMAGE_SIZE = 384
TEXT_LENGTH = 64
TEXT_BATCH = 32
ROW_CHUNK = 2048
MAX_BATCH = 64
LOADER_THREADS = 4
MIN_FREE_GB = 1
PROGRESS_INTERVAL = 5000
CACHE_QUALITY = 90
CACHE_FILL = (127, 127, 127)
PROMPTS = {
    "watermark_or_logo": ["a food photo with a visible stock photo watermark", "a food photo covered by a large translucent logo"],
    "text_poster_or_menu": ["a restaurant menu with prices and text", "a food advertisement poster with large writing"],
    "collage_or_grid": ["a collage of several different food photos", "a grid of multiple dish photographs"],
    "screenshot_or_ui": ["a screenshot of a food delivery app", "a screenshot of a web page showing food"],
    "drawing_or_illustration": ["an illustration of food", "a drawing of a cooked dish"],
    "not_food": ["a photograph of an object that is not food", "a photograph of a room or landscape"],
    "packaged_product": ["a packaged food product with printed label", "a food package photographed for a shop"],
    "person_only": ["a portrait of a person without visible food", "a person posing for a photograph"]
}

@dataclass
class ScoreContext:
    model: object
    processor: object
    category_names: list[str]
    category_vectors: torch.Tensor
    class_ids: list[str]
    class_vectors: torch.Tensor

def build_class_prompts() -> tuple[list[str], dict[str, list[str]]]:
    descriptions = {}
    search_names = {}
    search_path = REPO_ROOT_DIR / "notes" / "research" / "32-image-search-terms.csv"

    with search_path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            descriptions.setdefault(record["source_id"], record["visual_intent"].split(";")[0])
            search_names.setdefault(record["source_id"], record["item"].split(" / ")[0])

    class_ids = []
    prompts = {}

    for row in read_class_map(CLASS_MAP_PATH):
        if not row.source_id or row.source_id == "bukan_makanan":
            continue

        class_ids.append(row.source_id)
        names = [row.name_id]

        if row.name_en:
            names.append(row.name_en)

        phrases = [f"a photograph of {names[0]}", f"a plate of {names[0]}"]

        if row.source_id in descriptions:
            phrases.append(f"{names[0]}, {descriptions[row.source_id]}")

        prompts[row.source_id] = phrases

    for source_id in sorted(search_names):
        if source_id in prompts:
            continue

        class_ids.append(source_id)
        name = search_names[source_id]
        prompts[source_id] = [f"a photograph of {name}", f"a plate of {name}", f"{name}, {descriptions[source_id]}"]

    return class_ids, prompts

def normalize_features(features: torch.Tensor) -> torch.Tensor:
    if hasattr(features, "pooler_output"):
        features = features.pooler_output

    return torch.nn.functional.normalize(features.float(), dim=-1)

def encode_texts(model, processor, texts: list[str]) -> torch.Tensor:
    vectors = []

    for offset in range(0, len(texts), TEXT_BATCH):
        inputs = processor(
            text=texts[offset:offset + TEXT_BATCH],
            padding="max_length",
            truncation=True,
            max_length=TEXT_LENGTH,
            return_tensors="pt"
        ).to("cuda")

        with torch.inference_mode():
            features = model.get_text_features(input_ids=inputs["input_ids"])

        vectors.append(normalize_features(features))

    return torch.cat(vectors)

def build_text_vectors(model, processor) -> tuple[list[str], torch.Tensor, list[str], torch.Tensor]:
    category_names = list(PROMPTS)
    class_ids, class_prompts = build_class_prompts()
    category_vectors = []
    class_vectors = []
    all_prompts = []

    for category_name in category_names:
        all_prompts.extend(PROMPTS[category_name])

    for source_id in class_ids:
        all_prompts.extend(class_prompts[source_id])

    encoded = encode_texts(model, processor, all_prompts)
    offset = 0

    for category_name in category_names:
        prompt_count = len(PROMPTS[category_name])
        vectors = encoded[offset:offset + prompt_count]
        category_vectors.append(normalize_features(vectors.mean(dim=0, keepdim=True))[0])
        offset += prompt_count

    for source_id in class_ids:
        prompt_count = len(class_prompts[source_id])
        vectors = encoded[offset:offset + prompt_count]
        class_vectors.append(normalize_features(vectors.mean(dim=0, keepdim=True))[0])
        offset += prompt_count

    return category_names, torch.stack(category_vectors), class_ids, torch.stack(class_vectors)

def cache_image(row: dict, cache_dir: Path) -> Path:
    cache_path = cache_dir / f"{row['candidate_id']}.jpg"

    if cache_path.is_file():
        return cache_path

    temporary_path = cache_path.with_suffix(".tmp")

    with Image.open(row["local_path"]) as image:
        image.seek(0)
        upright = ImageOps.exif_transpose(image)
        rgb_image = convert_to_rgb(upright)
        padded = ImageOps.pad(rgb_image, (IMAGE_SIZE, IMAGE_SIZE), color=CACHE_FILL)
        padded.save(temporary_path, format="JPEG", quality=CACHE_QUALITY)

    temporary_path.replace(cache_path)
    return cache_path

def load_cached_image(row: dict, cache_dir: Path) -> Image.Image:
    with Image.open(cache_image(row, cache_dir)) as image:
        return image.convert("RGB")

def score_batch(context: ScoreContext, rows: list[dict], cache_dir: Path, loader: ThreadPoolExecutor) -> list[dict]:
    images = list(loader.map(partial(load_cached_image, cache_dir=cache_dir), rows))

    inputs = context.processor(images=images, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        features = context.model.get_image_features(pixel_values=inputs["pixel_values"])

    vectors = normalize_features(features)
    category_scores = (vectors @ context.category_vectors.T).cpu().numpy()
    class_scores = (vectors @ context.class_vectors.T).cpu().numpy()
    outputs = []
    class_positions = {source_id: position for position, source_id in enumerate(context.class_ids)}

    for position, row in enumerate(rows):
        output = {
            "candidate_id": row["candidate_id"],
            "source_id": row["source_id"],
            "discovery_route": row["discovery_route"],
            "query_id": row["query_id"],
            "local_path": row["local_path"],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "version": SCORE_VERSION,
            "embedding": vectors[position].cpu().numpy().astype(np.float16).tobytes()
        }

        for category_position, category_name in enumerate(context.category_names):
            output[category_name] = float(category_scores[position, category_position])

        class_position = class_positions.get(row["source_id"])

        if class_position is None:
            raise ValueError(f"missing class map entry for {row['source_id']}")

        given_score = class_scores[position, class_position]
        class_scores[position, class_position] = -1
        output["given_class_score"] = float(given_score)
        output["competing_class_score"] = float(class_scores[position].max())
        output["class_margin"] = output["given_class_score"] - output["competing_class_score"]
        outputs.append(output)

    return outputs

def score_rows(context: ScoreContext, rows: list[dict], cache_dir: Path) -> list[dict]:
    outputs = []
    offset = 0
    batch_size = MAX_BATCH

    with ThreadPoolExecutor(max_workers=LOADER_THREADS) as loader:
        while offset < len(rows):
            free_bytes = torch.cuda.mem_get_info()[0]

            if free_bytes < MIN_FREE_GB * 1024 ** 3:
                raise RuntimeError("less than 1 GB free GPU memory before scoring")

            batch = rows[offset:offset + batch_size]

            try:
                outputs.extend(score_batch(context, batch, cache_dir, loader))
                offset += len(batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_size //= 2
                logging.warning("reduced GPU batch to %s after out-of-memory", batch_size)

                if batch_size == 0:
                    raise RuntimeError("SigLIP2 cannot score one image within the GPU memory limit") from None

    return outputs

def score_part_path(rows: list[dict], parts_dir: Path) -> Path:
    digest = hashlib.sha256()
    digest.update(MODEL_REVISION.encode())
    digest.update(SCORE_VERSION.encode())

    for row in rows:
        digest.update(row["candidate_id"].encode())

    return parts_dir / f"{digest.hexdigest()[:20]}.parquet"

def run_score(data_root: Path, limit: int | None) -> tuple[Path, int, float]:
    probe_path = data_root / "acquisition" / "clean" / "probe.parquet"

    if not probe_path.is_file():
        raise FileNotFoundError(f"missing probe: {probe_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("SigLIP2 scoring requires a CUDA GPU")

    clean_dir = probe_path.parent
    parts_dir = clean_dir / "score_parts"
    cache_dir = data_root / "work" / "triage_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=data_root / "triage_score.log", level=logging.INFO, format="%(message)s", force=True)
    start_run("data-triage-score", {"model_id": MODEL_ID, "revision": MODEL_REVISION, "version": SCORE_VERSION})
    cache_path = data_root / "hf_cache" / "hub"
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_path, local_files_only=True, dtype=torch.float16).eval().cuda()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_path, local_files_only=True)
    category_names, category_vectors, class_ids, class_vectors = build_text_vectors(model, processor)
    context = ScoreContext(model, processor, category_names, category_vectors, class_ids, class_vectors)
    done = set()

    for existing_path in parts_dir.glob("*.parquet"):
        done.update(pl.read_parquet(existing_path, columns=["candidate_id"])["candidate_id"].to_list())

    started = time.monotonic()
    completed = 0
    scanned = 0

    for frame in pl.scan_parquet(probe_path).collect_batches(chunk_size=ROW_CHUNK):
        rows = []

        for row in frame.to_dicts():
            if row["hard_reasons"] or row["candidate_id"] in done:
                continue

            if limit is not None and scanned >= limit:
                break

            rows.append(row)
            scanned += 1

        if not rows:
            if limit is not None and scanned >= limit:
                break

            continue

        part_path = score_part_path(rows, parts_dir)

        if not part_path.is_file():
            outputs = score_rows(context, rows, cache_dir)
            temporary_path = part_path.with_suffix(".tmp")
            pl.DataFrame(outputs).write_parquet(temporary_path)
            temporary_path.replace(part_path)

        completed += len(rows)

        if completed // PROGRESS_INTERVAL > (completed - len(rows)) // PROGRESS_INTERVAL:
            logging.info("scored %s images in %.1f seconds", completed, time.monotonic() - started)

        if limit is not None and scanned >= limit:
            break

    parts = []

    for part_path in sorted(parts_dir.glob("*.parquet")):
        parts.append(pl.scan_parquet(part_path))

    output_path = clean_dir / "score.parquet"
    temporary_path = output_path.with_suffix(".tmp")
    pl.concat(parts).unique(subset=["candidate_id"], keep="last").sort("candidate_id").sink_parquet(temporary_path)
    temporary_path.replace(output_path)
    return output_path, completed, time.monotonic() - started

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye data triage score")
    parser.add_argument("--data-root", type=Path, default=get_data_root_dir())
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args(argv)
    output_path, count, seconds = run_score(arguments.data_root, arguments.limit)
    return [f"scored {count} images in {seconds:.1f}s", f"score {output_path}"]
