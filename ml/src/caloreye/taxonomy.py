import argparse
import csv
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from caloreye.paths import get_data_root_dir

REPO_ROOT_DIR = Path(__file__).resolve().parents[3]
CLASS_MAP_PATH = REPO_ROOT_DIR / "ml" / "contracts" / "class_map.csv"
NEEDS_LIST_PATH = REPO_ROOT_DIR / "notes" / "research" / "31-image-needs-list.md"
ADDITIONS_PATH = REPO_ROOT_DIR / "notes" / "research" / "31b-image-needs-additions.md"
IMAGES_RELATIVE_PATH = Path("acquisition") / "images"
NEEDS_LIST_START_MARKER = "## 2. The 100 items"
NEEDS_LIST_END_MARKER = "## 4. Look-alike"
TABLE_COLUMN_SEPARATOR = "|"
ID_COLUMN = 1
ITEM_COLUMN = 2
TIER_COLUMN = 3
PLAN_COLUMN = 4
WAVE_COLUMN = 8
NOTE_COLUMN = 9
MIN_TABLE_COLUMNS = 10
ROW_ID_PATTERN = re.compile(r"^(?:[0-9]+|[BN][0-9]+)$")
PARENTHESES_PATTERN = re.compile(r"\(([^)]*)\)")
PILOT_SOURCE_IDS = ("1", "5", "6", "11", "39", "B1")
NOT_FOOD_SOURCE_ID = "bukan_makanan"
NOT_FOOD_NAME_ID = "Bukan makanan"
NOT_FOOD_NAME_EN = "Not food"
NOT_FOOD_NOTE = "reserved stage two class, no image folder"
EXPECTED_PILOT_COUNT = 6
EXPECTED_ROW_COUNT = 268

@dataclass
class ClassRow:
    source_id: str
    name_id: str
    name_en: str
    tier: str
    plan: str
    wave: str
    stage2_class: bool
    pilot: bool
    notes: str

def split_item_names(item_text: str) -> tuple[str, str]:
    english_parts = PARENTHESES_PATTERN.findall(item_text)
    indonesian_name = PARENTHESES_PATTERN.sub("", item_text).strip()
    english_name = "; ".join(english_parts).strip()

    return indonesian_name, english_name

def parse_table_row(line: str) -> ClassRow | None:
    cells = [cell.strip() for cell in line.split(TABLE_COLUMN_SEPARATOR)]

    if len(cells) < MIN_TABLE_COLUMNS:
        return None

    source_id = cells[ID_COLUMN]

    if not ROW_ID_PATTERN.match(source_id):
        return None

    name_id, name_en = split_item_names(cells[ITEM_COLUMN])
    is_pilot = source_id in PILOT_SOURCE_IDS

    return ClassRow(
        source_id=source_id,
        name_id=name_id,
        name_en=name_en,
        tier=cells[TIER_COLUMN],
        plan=cells[PLAN_COLUMN],
        wave=cells[WAVE_COLUMN],
        stage2_class=is_pilot,
        pilot=is_pilot,
        notes=cells[NOTE_COLUMN]
    )

def parse_needs_list(markdown_text: str) -> list[ClassRow]:
    rows = []
    start_index = markdown_text.find(NEEDS_LIST_START_MARKER)
    end_index = markdown_text.find(NEEDS_LIST_END_MARKER)
    section_text = markdown_text[start_index:end_index]

    for line in section_text.splitlines():
        row = parse_table_row(line)

        if row is not None:
            rows.append(row)

    return rows

def parse_additions(markdown_text: str) -> list[ClassRow]:
    rows = []

    for line in markdown_text.splitlines():
        row = parse_table_row(line)

        if row is not None:
            rows.append(row)

    return rows

def build_class_rows(needs_text: str, additions_text: str) -> list[ClassRow]:
    rows = parse_needs_list(needs_text) + parse_additions(additions_text)
    rows.append(
        ClassRow(
            source_id=NOT_FOOD_SOURCE_ID,
            name_id=NOT_FOOD_NAME_ID,
            name_en=NOT_FOOD_NAME_EN,
            tier="",
            plan="",
            wave="",
            stage2_class=False,
            pilot=False,
            notes=NOT_FOOD_NOTE
        )
    )

    return rows

def write_class_map(rows: list[ClassRow], output_path: Path) -> None:
    column_names = [field.name for field in fields(ClassRow)]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=column_names, lineterminator="\n")
        writer.writeheader()

        for row in rows:
            record = asdict(row)
            record["stage2_class"] = str(row.stage2_class).lower()
            record["pilot"] = str(row.pilot).lower()
            writer.writerow(record)

def read_class_map(input_path: Path) -> list[ClassRow]:
    rows = []

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                ClassRow(
                    source_id=record["source_id"],
                    name_id=record["name_id"],
                    name_en=record["name_en"],
                    tier=record["tier"],
                    plan=record["plan"],
                    wave=record["wave"],
                    stage2_class=record["stage2_class"] == "true",
                    pilot=record["pilot"] == "true",
                    notes=record["notes"]
                )
            )

    return rows

def list_image_folder_names(images_dir: Path) -> list[str]:
    if not images_dir.is_dir():
        return []

    folder_names = []

    for child_path in images_dir.iterdir():
        if child_path.is_dir():
            folder_names.append(child_path.name)

    return sorted(folder_names)

def find_problems(
        rows: list[ClassRow],
        folder_names: list[str],
        expected_row_count: int | None = EXPECTED_ROW_COUNT
    ) -> list[str]:
    problems = []
    source_ids = [row.source_id for row in rows]
    seen_ids = set()

    for source_id in source_ids:
        if source_id in seen_ids:
            problems.append(f"duplicate source_id {source_id}")

        seen_ids.add(source_id)

    pilot_ids = sorted([row.source_id for row in rows if row.pilot])
    expected_pilot_ids = sorted(PILOT_SOURCE_IDS)

    if pilot_ids != expected_pilot_ids:
        problems.append(f"expected pilot rows {expected_pilot_ids}, found {pilot_ids}")

    stage2_ids = sorted([row.source_id for row in rows if row.stage2_class])

    if stage2_ids != expected_pilot_ids:
        problems.append(f"expected stage two rows {expected_pilot_ids}, found {stage2_ids}")

    if expected_row_count is not None and len(rows) != expected_row_count:
        problems.append(f"expected {expected_row_count} rows, found {len(rows)}")

    for folder_name in folder_names:
        if folder_name not in seen_ids:
            problems.append(f"image folder {folder_name} is not in the class map")

    return problems

def run_check(class_map_path: Path, images_dir: Path) -> tuple[int, list[str]]:
    rows = read_class_map(class_map_path)
    folder_names = list_image_folder_names(images_dir)
    problems = find_problems(rows, folder_names)
    pilot_count = len([row for row in rows if row.pilot])
    lines = [f"rows {len(rows)}  pilot {pilot_count}  image folders {len(folder_names)}"]

    for problem in problems:
        lines.append(f"PROBLEM {problem}")

    if problems:
        return 1, lines

    lines.append("taxonomy check passed")

    return 0, lines

def run_build(needs_path: Path, additions_path: Path, output_path: Path) -> tuple[int, list[str]]:
    needs_text = needs_path.read_text(encoding="utf-8")
    additions_text = additions_path.read_text(encoding="utf-8")
    rows = build_class_rows(needs_text, additions_text)
    write_class_map(rows, output_path)

    return 0, [f"wrote {len(rows)} rows to {output_path}"]

def main(argv: list[str] | None = None) -> tuple[int, list[str]]:
    parser = argparse.ArgumentParser(prog="caloreye taxonomy")
    parser.add_argument("action", choices=["check", "build"])
    parser.add_argument("--class-map", type=Path, default=CLASS_MAP_PATH)
    parser.add_argument("--images-dir", type=Path)
    arguments = parser.parse_args(argv)
    images_dir = arguments.images_dir

    if images_dir is None:
        images_dir = get_data_root_dir() / IMAGES_RELATIVE_PATH

    if arguments.action == "build":
        return run_build(NEEDS_LIST_PATH, ADDITIONS_PATH, arguments.class_map)

    return run_check(arguments.class_map, images_dir)
