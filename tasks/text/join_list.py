"""Join a list of strings with a separator."""

from core.task import task


@task(
    outputs=["result"],
    display_name="Join List",
    description="Join a list of strings with a separator",
    category="text",
    output_types={"result": "text"},
    is_collapsed=True,
    parameters={
        "items": {
            "type": "list",
            "required": True,
            "description": "List of items to join",
        },
        "separator": {
            "type": "str",
            "required": False,
            "default": "\n",
            "description": "String to insert between items",
        },
    },
)
def join_list(items: list, separator: str = "\n") -> str:
    """
    Join a list of strings with a separator.

    Args:
        items: List of items to join (will be converted to strings)
        separator: String to insert between items

    Returns:
        Joined string
    """
    if items is None:
        return ""
    return separator.join(str(item) for item in items)
