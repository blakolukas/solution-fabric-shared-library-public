import os
import uuid

from core.task import task


@task(
    outputs=["temp_filename"],
    display_name="Generate Temp Filename",
    description="Generate a unique temporary filename",
    category="io",
    parameters={
        "prefix": {
            "type": "str",
            "required": False,
            "default": "temp",
            "description": "Prefix for the filename",
        },
        "extension": {
            "type": "str",
            "required": False,
            "default": "png",
            "description": "File extension without dot",
        },
    },
)
def generate_temp_filename(prefix: str = "temp", extension: str = "png"):
    """
    Generate a unique temporary filename.

    Args:
        prefix: Prefix for the filename (default: "temp")
        extension: File extension without dot (default: "png")

    Returns:
        Unique filename with extension
    """
    return f"{prefix}_{uuid.uuid4().hex}.{extension}"
