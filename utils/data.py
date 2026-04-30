"""CSV/JSON serialization helpers for dataset and prediction I/O."""
import csv
import json

from pathlib import Path

from pandas import Series


def cast_to_json(column: Series) -> Series:
    return column.apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
    )


def infer_json(column: Series) -> Series:
    return column.apply(_parse_json, column_name=column.name)


def _parse_json(value, column_name):
    """Attempt JSON parse on columns known to contain serialized structures."""
    if any(v in column_name for v in ["span", "prediction", "techniques", "result", "attack_target"]):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def csv_batch_writer(batch: list[dict], filename: str | Path) -> None:
    """Append a single batch of dicts to a CSV, writing a header on first call."""
    if not batch:
        return
    filepath = Path(filename)
    file_exists = filepath.is_file() and filepath.stat().st_size > 0
    fieldnames = batch[0].keys()
    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(batch)
