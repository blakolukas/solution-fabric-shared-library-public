"""Split text into lines."""

from typing import List

from core.task import task


@task(
    outputs=["lines"],
    display_name="Split Lines",
    description="Split text into lines",
    category="text",
    output_types={"lines": "list"},
    is_collapsed=True,
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to split",
        },
        "skip_empty": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "Skip empty lines",
        },
    },
)
def split_lines(text: str, skip_empty: bool = True) -> List[str]:
    """
    Split text into lines.

    Args:
        text: Text to split
        skip_empty: Skip empty lines

    Returns:
        List of lines
    """
    if text is None:
        return []

    lines = str(text).splitlines()

    if skip_empty:
        lines = [line for line in lines if line.strip()]

    return lines
