import hashlib
import json
import logging
import random
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

from caloreye.paths import REPO_ROOT_DIR, get_data_root_dir, get_runs_dir

HASH_CHUNK_BYTES = 1024 * 1024
EMPTY_BYTES = b""
RUN_NAME_PATTERN = r"[^a-zA-Z0-9_-]+"
RUN_ID_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
RUN_ID_SUFFIX_LENGTH = 8
RUN_RECORD_FILENAME = "run.json"
MANIFEST_RELATIVE_PATH = Path("work") / "manifest.parquet"
UV_LOCK_RELATIVE_PATH = Path("ml") / "uv.lock"
GIT_STATUS_COMMAND = ("git", "status", "--porcelain")
GIT_REVISION_COMMAND = ("git", "rev-parse", "HEAD")
UV_VERSION_COMMAND = ("uv", "--version")
CUDA_SEED_WARNING = "Seed %d set; CUDA kernels are not fully deterministic"
JSON_INDENT = 2

@dataclass(frozen=True)
class RunDir:
    run_dir: Path

def hash_file(file_path: Path) -> str | None:
    if not file_path.is_file():
        return None

    digest = hashlib.sha256()

    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(HASH_CHUNK_BYTES), EMPTY_BYTES):
            digest.update(chunk)

    return digest.hexdigest()

def run_command(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            cwd=REPO_ROOT_DIR,
            capture_output=True,
            text=True,
            check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return completed.stdout.strip()

def read_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logging.warning(CUDA_SEED_WARNING, seed)

def create_run_id(run_name: str) -> str:
    safe_name = re.sub(RUN_NAME_PATTERN, "-", run_name).strip("-")

    if not safe_name:
        raise ValueError("Run name must contain a letter or digit")

    started_at = datetime.now(UTC).strftime(RUN_ID_TIME_FORMAT)
    unique_suffix = uuid.uuid4().hex[:RUN_ID_SUFFIX_LENGTH]

    return f"{started_at}-{safe_name}-{unique_suffix}"

def read_git_dirty_status() -> bool | None:
    status = run_command(*GIT_STATUS_COMMAND)

    if status is None:
        return None

    return bool(status)

def collect_run_versions() -> dict[str, str | None]:
    return {
        "torch": torch.__version__,
        "ultralytics": read_package_version("ultralytics"),
        "executorch": read_package_version("executorch"),
        "timm": read_package_version("timm")
    }

def build_run_record(run_id: str, run_name: str, config: dict) -> dict:
    return {
        "run_id": run_id,
        "name": run_name,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "seed": config.get("seed"),
        "git_sha": run_command(*GIT_REVISION_COMMAND),
        "git_dirty": read_git_dirty_status(),
        "uv_version": run_command(*UV_VERSION_COMMAND),
        "versions": collect_run_versions(),
        "manifest_sha256": hash_file(get_data_root_dir() / MANIFEST_RELATIVE_PATH),
        "uv_lock_sha256": hash_file(REPO_ROOT_DIR / UV_LOCK_RELATIVE_PATH)
    }

def start_run(name: str, config: dict) -> RunDir:
    run_id = create_run_id(name)
    run_dir = get_runs_dir(create=True) / run_id
    run_dir.mkdir()

    record = build_run_record(run_id, name, config)
    record_json = json.dumps(record, indent=JSON_INDENT) + "\n"
    record_path = run_dir / RUN_RECORD_FILENAME
    record_path.write_text(record_json, encoding="utf-8")

    return RunDir(run_dir)
