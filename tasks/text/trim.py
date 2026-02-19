"""Trim whitespace and optionally transform text."""

from core.task import task


@task(
    outputs=["trimmed"],
    display_name="Trim",
    description="Trim whitespace and optionally transform text",
    category="text",
    output_types={"trimmed": "str"},
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to trim",
        },
        "lowercase": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Convert to lowercase after trimming",
        },
        "uppercase": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Convert to uppercase after trimming",
        },
    },
)
def trim(
    text: str,
    lowercase: bool = False,
    uppercase: bool = False,
) -> str:
    """
    Trim whitespace from text and optionally transform case.

    Args:
        text: Text to trim
        lowercase: Convert to lowercase after trimming
        uppercase: Convert to uppercase after trimming

    Returns:
        Trimmed (and optionally transformed) text
    """
    if text is None:
        return ""

    result = str(text).strip()

    if lowercase:
        result = result.lower()
    elif uppercase:
        result = result.upper()

    return result
