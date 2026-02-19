"""Extract a substring from text."""

from typing import Optional

from core.task import task


@task(
    outputs=["substring"],
    display_name="Substring",
    description="Extract a substring from text",
    category="text",
    output_types={"substring": "str"},
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to extract from",
        },
        "start": {
            "type": "int",
            "required": False,
            "default": 0,
            "description": "Start index (0-based)",
        },
        "end": {
            "type": "int",
            "required": False,
            "default": None,
            "description": "End index (exclusive), None = to end",
        },
    },
)
def substring(
    text: str,
    start: int = 0,
    end: Optional[int] = None,
) -> str:
    """
    Extract a substring from text.

    Args:
        text: Text to extract from
        start: Start index (0-based)
        end: End index (exclusive), None = to end

    Returns:
        Extracted substring
    """
    if text is None:
        return ""

    if end is None:
        return str(text)[start:]
    else:
        return str(text)[start:end]
