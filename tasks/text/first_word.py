"""Extract the first word from text."""

from core.task import task


@task(
    outputs=["word"],
    display_name="First Word",
    description="Extract the first word from text",
    category="text",
    output_types={"word": "text"},
    is_collapsed=True,
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to extract first word from",
        },
        "lowercase": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "Convert result to lowercase",
        },
    },
)
def first_word(text: str, lowercase: bool = True) -> str:
    """
    Extract the first word from text.

    Useful for parsing single-word LLM responses like "yes", "no", "relevant".

    Args:
        text: Text to extract from
        lowercase: Convert result to lowercase (default: True)

    Returns:
        First word found, or empty string if none
    """
    if text is None:
        return ""

    # Split on whitespace and get first non-empty word
    words = str(text).split()
    if not words:
        return ""

    result = words[0]

    # Remove common punctuation from the word
    result = result.strip(".,!?;:'\"")

    if lowercase:
        result = result.lower()

    return result
