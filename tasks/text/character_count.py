"""Count the number of characters in text."""

from core.task import task


@task(
    outputs=["char_count"],
    display_name="Character Count",
    description="Count the number of characters in text",
    category="text",
    output_types={"char_count": "int"},
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to count",
        },
        "exclude_spaces": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "If True, don't count whitespace",
        },
    },
)
def char_count(text: str, exclude_spaces: bool = False) -> int:
    """
    Count the number of characters in text.

    Args:
        text: Text to count
        exclude_spaces: If True, don't count whitespace

    Returns:
        Number of characters
    """
    if text is None:
        return 0

    text_str = str(text)
    if exclude_spaces:
        text_str = text_str.replace(" ", "").replace("\t", "").replace("\n", "")

    return len(text_str)
