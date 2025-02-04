import json

from pandas import Series


def cast_to_json(column: Series) -> Series:
    return column.apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)


def infer_json(column: Series) -> Series:
    return column.apply(parse_json, column_name=column.name)


def parse_json(value, column_name):
    if "span" in column_name or "techniques" in column_name:  # TODO: improve
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value
