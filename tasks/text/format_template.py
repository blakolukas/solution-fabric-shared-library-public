"""Format a template string by replacing placeholders with values."""

import re
from typing import Any, Dict, Optional

from core.task import task


@task(
    outputs=["result"],
    display_name="Format Template",
    description="Format a template string by replacing placeholders with values",
    category="text",
    output_types={"result": "text"},
    is_collapsed=True,
    parameters={
        "template": {
            "type": "str",
            "required": True,
            "dynamicInputSource": True,
            "description": "Template string with {variable} placeholders",
        },
        "values": {
            "type": "dict",
            "required": False,
            "default": None,
            "description": "Dictionary of key-value pairs for named placeholders",
        },
    },
)
def format_template(template: str, values: Optional[Dict[str, Any]] = None, **kwargs) -> str:
    """
    Format a template string by replacing placeholders with values.

    Supports two placeholder styles:
    - Python format: {key} or {0}, {1} for positional
    - Dollar style: $key or ${key}

    Dynamic variables detected from the template (like {name}, {count}) are
    automatically available as connection inputs in the Designer UI.

    Args:
        template: Template string with placeholders
        values: Dictionary of key-value pairs for named placeholders
        **kwargs: Dynamic variables detected from template placeholders

    Returns:
        Formatted string with placeholders replaced

    Example:
        template="Hello {name}, you have {count} messages"
        -> Creates dynamic inputs for 'name' and 'count'
    """
    if template is None:
        return ""

    # Merge values dict with kwargs (dynamic variables from template)
    all_values = dict(values) if values else {}
    for key, value in kwargs.items():
        if value is not None:
            all_values[key] = value

    result = template

    # Handle $key and ${key} style placeholders first
    def dollar_replace(match):
        key = match.group(1) or match.group(2)
        return str(all_values.get(key, match.group(0)))

    result = re.sub(r"\$\{(\w+)\}|\$(\w+)", dollar_replace, result)

    # Handle Python {key} style placeholders
    try:
        result = result.format(**all_values)
    except KeyError:
        # If some keys are missing, do partial formatting
        for key, value in all_values.items():
            result = result.replace(f"{{{key}}}", str(value))

    return result
