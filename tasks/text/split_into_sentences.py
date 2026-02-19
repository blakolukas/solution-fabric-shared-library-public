"""Split text into sentences."""

import re
from typing import List

from core.task import task


@task(
    outputs=["sentences"],
    display_name="Split Into Sentences",
    description="Split text into sentences",
    category="text",
    output_types={"sentences": "list"},
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to split",
        },
    },
)
def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Uses simple regex-based sentence splitting.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    if text is None or not text.strip():
        return []

    # Simple sentence splitting on common sentence terminators
    # This handles . ! ? followed by space or end of string
    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())

    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]
