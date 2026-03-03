"""Concatenate multiple text values with a separator."""

from typing import Optional

from core.task import task


@task(
    outputs=["result"],
    display_name="Concatenate",
    description="Concatenate multiple text values with a separator",
    category="text",
    output_types={"result": "text"},
    is_collapsed=True,
    parameters={
        "text_1": {
            "type": "str",
            "required": False,
            "default": None,
            "description": "First text value",
        },
        "text_2": {
            "type": "str",
            "required": False,
            "default": None,
            "description": "Second text value",
        },
        "text_3": {
            "type": "str",
            "required": False,
            "default": None,
            "description": "Third text value",
        },
        "text_4": {
            "type": "str",
            "required": False,
            "default": None,
            "description": "Fourth text value",
        },
        "text_5": {
            "type": "str",
            "required": False,
            "default": None,
            "description": "Fifth text value",
        },
        "text_6": {
            "type": "str",
            "required": False,
            "default": None,
            "description": "Sixth text value",
        },
        "separator": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "String to insert between values",
        },
        "skip_empty": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "If True, skip None and empty string values",
        },
    },
)
def concatenate(
    text_1: Optional[str] = None,
    text_2: Optional[str] = None,
    text_3: Optional[str] = None,
    text_4: Optional[str] = None,
    text_5: Optional[str] = None,
    text_6: Optional[str] = None,
    separator: str = "",
    skip_empty: bool = True,
) -> str:
    """
    Concatenate multiple text values with a separator.

    Args:
        text_1 through text_6: Text values to concatenate
        separator: String to insert between values (default: no separator)
        skip_empty: If True, skip None and empty string values

    Returns:
        Concatenated string
    """
    texts = [text_1, text_2, text_3, text_4, text_5, text_6]

    if skip_empty:
        texts = [t for t in texts if t is not None and str(t).strip()]
    else:
        texts = [str(t) if t is not None else "" for t in texts]

    return separator.join(str(t) for t in texts)
