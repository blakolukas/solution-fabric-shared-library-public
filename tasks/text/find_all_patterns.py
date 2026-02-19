"""Find all occurrences matching a regex pattern."""

import re
from typing import List

from core.task import task


@task(
    outputs=["matches"],
    display_name="Find All Patterns",
    description="Find all occurrences matching a regex pattern",
    category="text",
    output_types={"matches": "list"},
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
    },
)
def find_all_patterns(text: str, pattern: str) -> List[str]:
    """
    Find all occurrences matching a regex pattern.

    Args:
        text: Text to search in
        pattern: Regular expression pattern

    Returns:
        List of all matches
    """
    if text is None:
        return []

    try:
        return re.findall(pattern, str(text))
    except re.error:
        return []
