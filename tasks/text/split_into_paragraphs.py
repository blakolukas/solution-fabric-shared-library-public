"""Split text into paragraphs."""

import re
from typing import List

from core.task import task


@task(
    outputs=["paragraphs"],
    display_name="Split Into Paragraphs",
    description="Split text into paragraphs",
    category="text",
    output_types={"paragraphs": "list"},
    is_collapsed=True,
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to split",
        },
    },
)
def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs.

    Paragraphs are separated by one or more blank lines.

    Args:
        text: Text to split

    Returns:
        List of paragraphs
    """
    if text is None or not text.strip():
        return []

    # Split on one or more blank lines
    paragraphs = re.split(r"\n\s*\n", str(text).strip())

    # Filter out empty paragraphs and strip whitespace
    return [p.strip() for p in paragraphs if p.strip()]
