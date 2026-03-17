"""Parse a JSON string into a Python dictionary or list."""

import json

from core.task import task


@task(
    outputs=["data"],
    display_name="Parse JSON",
    description="Parse a JSON string into a Python dictionary or list",
    category="text",
    output_types={"data": "json"},
    parameters={
        "json_string": {
            "type": "str",
            "required": True,
            "description": "Valid JSON string to parse",
        },
    },
)
def parse_json(json_string: str):
    """
    Parse a JSON string into a Python dictionary or list.

    Args:
        json_string: Valid JSON string to parse

    Returns:
        data: Parsed Python object (dict or list)

    Raises:
        ValueError: If the input is not valid JSON
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
