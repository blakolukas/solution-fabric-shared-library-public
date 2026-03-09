"""Extract text matching a regex pattern."""

import re

from core.task import task


@task(
    outputs=["match"],
    display_name="Extract Pattern",
    description="Extract text matching a regex pattern",
    category="text",
    output_types={"match": "text"},
    is_collapsed=True,
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to search in",
        },
        "pattern": {
            "type": "str",
            "required": True,
            "description": "Regular expression pattern",
        },
        "group": {
            "type": "int",
            "required": False,
            "default": 0,
            "description": "Capture group to return (0 = entire match)",
        },
        "default": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "Default value if no match found",
        },
    },
)
def extract_pattern(
    text: str,
    pattern: str,
    group: int = 0,
    default: str = "",
) -> str:
    """
    Extract text matching a regex pattern.

    Args:
        text: Text to search in
        pattern: Regular expression pattern
        group: Capture group to return (0 = entire match)
        default: Default value if no match found

    Returns:
        Matched text or default value
    """
    if text is None:
        return default

    try:
        match = re.search(pattern, str(text))
        if match:
            return match.group(group)
    except re.error:
        pass

    return default
