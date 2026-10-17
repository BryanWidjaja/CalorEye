import argparse
import shutil
import sys
from pathlib import Path

import torch

from caloreye.paths import get_data_root_dir
from caloreye.runs import read_package_version

ACQUISITION_IMAGES_RELATIVE_PATH = Path("acquisition") / "images"
SAM3_CACHE_RELATIVE_PATH = Path(".cache/huggingface/hub/models--facebook--sam3/snapshots")
SAM3_WEIGHTS_PATTERN = "*/sam3.pt"
BYTES_PER_GIB = 1024 ** 3
PACKAGE_NOT_INSTALLED = "not installed"
PRIMARY_CUDA_DEVICE_INDEX = 0
PYTHON_VERSION_COMPONENT_INDEX = 0

def display_package_version(package_name: str) -> str:
    package_version = read_package_version(package_name)

    if package_version is None:
        return PACKAGE_NOT_INSTALLED

    return package_version

def find_free_disk_path(root_path: Path) -> Path:
    for candidate_path in (root_path, *root_path.parents):
        if candidate_path.exists():
            return candidate_path

    return Path.cwd()

def count_acquisition_entries(images_dir: Path) -> tuple[int, int]:
    if not images_dir.is_dir():
        return (0, 0)

    folder_count = 0
    image_count = 0

    for source_dir in images_dir.iterdir():
        if source_dir.is_dir():
            folder_count += 1

    for image_path in images_dir.rglob("*"):
        if image_path.is_file():
            image_count += 1

    return image_count, folder_count

def has_sam3_weights() -> bool:
    cache_dir = Path.home() / SAM3_CACHE_RELATIVE_PATH

    if not cache_dir.is_dir():
        return False

    for weights_path in cache_dir.glob(SAM3_WEIGHTS_PATTERN):
        if weights_path.is_file():
            return True

    return False

def describe_gpu() -> tuple[str, str, str]:
    if not torch.cuda.is_available():
        return ("unavailable", "none", "n/a")

    gpu_name = torch.cuda.get_device_name(PRIMARY_CUDA_DEVICE_INDEX)
    gpu_properties = torch.cuda.get_device_properties(PRIMARY_CUDA_DEVICE_INDEX)
    vram_gib = gpu_properties.total_memory / BYTES_PER_GIB

    return "available", gpu_name, f"{vram_gib:.2f} GiB"

def collect_doctor_status() -> dict[str, str]:
    root_path = get_data_root_dir()
    images_dir = root_path / ACQUISITION_IMAGES_RELATIVE_PATH
    image_count, folder_count = count_acquisition_entries(images_dir)
    disk_path = find_free_disk_path(root_path)
    free_disk_gib = shutil.disk_usage(disk_path).free / BYTES_PER_GIB
    cuda_status, gpu_name, vram_size = describe_gpu()

    if has_sam3_weights():
        sam3_status = "found"
    else:
        sam3_status = "not found"

    return {
        "Python": sys.version.split()[PYTHON_VERSION_COMPONENT_INDEX],
        "Torch": torch.__version__,
        "CUDA": cuda_status,
        "GPU": gpu_name,
        "VRAM": vram_size,
        "Ultralytics": display_package_version("ultralytics"),
        "ExecuTorch": display_package_version("executorch"),
        "timm": display_package_version("timm"),
        "Free disk": f"{free_disk_gib:.2f} GiB",
        "Data root": str(root_path),
        "Acquisition images": str(image_count),
        "Acquisition folders": str(folder_count),
        "SAM 3 weights": sam3_status
    }

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="caloreye")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="show environment and acquisition status")
    export_command = commands.add_parser("export", help="export and verify models")
    export_actions = export_command.add_subparsers(dest="export_action", required=True)
    export_actions.add_parser("spike", help="export both models")
    export_actions.add_parser("parity", help="compare exported outputs and decoded masks")
    export_actions.add_parser("bundle", help="write validated manifest example")
    commands.add_parser("goldens", help="generate synthetic preprocessing vectors")
    taxonomy_command = commands.add_parser("taxonomy", help="class map build and check")
    taxonomy_command.add_argument("taxonomy_action", choices=["check", "build"])
    arguments, remaining = parser.parse_known_args(argv)

    if arguments.command == "doctor":
        for label, value in collect_doctor_status().items():
            print(f"{label:<22} {value}")

    if arguments.command == "goldens":
        from caloreye.export.goldens import main as generate_goldens

        generate_goldens(remaining)

    if arguments.command == "taxonomy":
        from caloreye.taxonomy import main as run_taxonomy

        exit_code, report_lines = run_taxonomy([arguments.taxonomy_action, *remaining])

        for report_line in report_lines:
            print(report_line)

        raise SystemExit(exit_code)

    if arguments.command == "export":
        if arguments.export_action == "spike":
            from caloreye.export.spike import main as export_spike

            export_spike(remaining)

        if arguments.export_action == "parity":
            from caloreye.export.parity import main as check_parity

            check_parity(remaining)

        if arguments.export_action == "bundle":
            from caloreye.export.bundle import main as write_bundle

            write_bundle(remaining)

if __name__ == "__main__":
    main()
