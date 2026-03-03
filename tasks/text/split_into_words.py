"""Split text into words."""

from typing import List

from core.task import task


@task(
    outputs=["words"],
    display_name="Split Into Words",
    description="Split text into words",
    category="text",
    output_types={"words": "list"},
    is_collapsed=True,
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to split",
        },
        "lowercase": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Convert words to lowercase",
        },
    },
)
def split_into_words(text: str, lowercase: bool = False) -> List[str]:
    """
    Split text into words.

    Args:
        text: Text to split
        lowercase: Convert words to lowercase

    Returns:
        List of words
    """
    if text is None or not text.strip():
        return []

    words = str(text).split()

    if lowercase:
        words = [w.lower() for w in words]

    return words
