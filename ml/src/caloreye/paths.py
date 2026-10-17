import os
from pathlib import Path

REPO_ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR_NAME = "data"
DATA_ROOT_ENV_NAME = "CALOREYE_DATA"
RUNS_DIR_NAME = "runs"

def get_data_root_dir() -> Path:
    configured_path = os.environ.get(DATA_ROOT_ENV_NAME, REPO_ROOT_DIR / DEFAULT_DATA_DIR_NAME)

    return Path(configured_path).expanduser().resolve()

def get_runs_dir(*, create: bool = False) -> Path:
    run_dir = get_data_root_dir() / RUNS_DIR_NAME

    if create:
        run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir
