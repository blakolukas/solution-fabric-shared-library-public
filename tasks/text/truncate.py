"""Truncate text to a maximum length."""

from core.task import task


@task(
    outputs=["truncated"],
    display_name="Truncate",
    description="Truncate text to a maximum length",
    category="text",
    output_types={"truncated": "str"},
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to truncate",
        },
        "max_length": {
            "type": "int",
            "required": False,
            "default": 100,
            "description": "Maximum length (including suffix)",
        },
        "suffix": {
            "type": "str",
            "required": False,
            "default": "...",
            "description": "Suffix to add if truncated",
        },
    },
)
def truncate(
    text: str,
    max_length: int = 100,
    suffix: str = "...",
) -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length (including suffix)
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if text is None:
        return ""

    text_str = str(text)
    if len(text_str) <= max_length:
        return text_str

    return text_str[: max_length - len(suffix)] + suffix
