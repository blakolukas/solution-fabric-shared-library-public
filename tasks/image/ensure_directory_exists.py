import os

from core.task import task


@task(
    outputs=["directory_path"],
    display_name="Ensure Directory Exists",
    description="Ensure a directory exists, creating it if necessary",
    category="io",
    parameters={
        "directory_path": {
            "type": "str",
            "required": True,
            "description": "Path to the directory",
        },
    },
)
def ensure_directory_exists(directory_path: str):
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory

    Returns:
        Absolute path to the directory
    """
    abs_path = os.path.abspath(directory_path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path
