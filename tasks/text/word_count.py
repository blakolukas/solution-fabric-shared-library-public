"""Count the number of words in text."""

from core.task import task


@task(
    outputs=["word_count"],
    display_name="Word Count",
    description="Count the number of words in text",
    category="text",
    output_types={"word_count": "int"},
    is_collapsed=True,
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to count",
        },
    },
)
def word_count(text: str) -> int:
    """
    Count the number of words in text.

    Args:
        text: Text to count

    Returns:
        Number of words
    """
    if text is None or not text.strip():
        return 0

    return len(str(text).split())
