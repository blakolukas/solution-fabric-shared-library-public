"""Extract the last word from text."""

from core.task import task


@task(
    outputs=["word"],
    display_name="Last Word",
    description="Extract the last word from text",
    category="text",
    output_types={"word": "str"},
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to extract last word from",
        },
        "lowercase": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "Convert result to lowercase",
        },
    },
)
def last_word(text: str, lowercase: bool = True) -> str:
    """
    Extract the last word from text.

    Args:
        text: Text to extract from
        lowercase: Convert result to lowercase

    Returns:
        Last word found, or empty string if none
    """
    if text is None:
        return ""

    words = str(text).split()
    if not words:
        return ""

    result = words[-1].strip(".,!?;:'\"")

    if lowercase:
        result = result.lower()

    return result
