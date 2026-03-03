"""Split text by a delimiter."""

from typing import List

from core.task import task


@task(
    outputs=["parts"],
    display_name="Split",
    description="Split text by a delimiter",
    category="text",
    output_types={"parts": "list"},
    is_collapsed=True,
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to split",
        },
        "delimiter": {
            "type": "str",
            "required": False,
            "default": ",",
            "description": "Delimiter to split on",
        },
        "max_splits": {
            "type": "int",
            "required": False,
            "default": -1,
            "description": "Maximum number of splits (-1 = unlimited)",
        },
    },
)
def split(text: str, delimiter: str = ",", max_splits: int = -1) -> List[str]:
    """
    Split text by a delimiter.

    Args:
        text: Text to split
        delimiter: Delimiter to split on
        max_splits: Maximum number of splits (-1 = unlimited)

    Returns:
        List of parts
    """
    if text is None:
        return []

    if max_splits >= 0:
        return str(text).split(delimiter, max_splits)
    else:
        return str(text).split(delimiter)
