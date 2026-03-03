"""Replace occurrences of a pattern in text."""

import re

from core.task import task


@task(
    outputs=["replaced"],
    display_name="Replace",
    description="Replace occurrences of a pattern in text",
    category="text",
    output_types={"replaced": "str"},
    is_collapsed=True,
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to process",
        },
        "find": {
            "type": "str",
            "required": True,
            "description": "String or pattern to find",
        },
        "replace_with": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "Replacement string",
        },
        "use_regex": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Treat 'find' as a regex pattern",
        },
    },
)
def replace(
    text: str,
    find: str,
    replace_with: str = "",
    use_regex: bool = False,
) -> str:
    """
    Replace occurrences of a pattern in text.

    Args:
        text: Text to process
        find: String or pattern to find
        replace_with: Replacement string
        use_regex: Treat 'find' as a regex pattern

    Returns:
        Text with replacements applied
    """
    if text is None:
        return ""

    if use_regex:
        try:
            return re.sub(find, replace_with, str(text))
        except re.error:
            return str(text)
    else:
        return str(text).replace(find, replace_with)
