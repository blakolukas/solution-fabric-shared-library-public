"""Check if a file exists and return its absolute path."""

import os

from core.task import task


@task(
    outputs=["file_path"],
    output_types={"file_path": "str"},
    display_name="Check File Exists",
    description="Validate that a file exists and return its absolute path",
    category="io",
    is_collapsed=True,
    parameters={
        "file_path": {
            "type": "str",
            "required": False,
            "description": "Path to the file to check. Returns None if not provided.",
        },
    },
)
def check_file_exists(file_path: str = None) -> str:
    """
    Check if a file exists and return its absolute path.

    This task validates that a file exists on the filesystem
    before it is processed by downstream tasks.

    Args:
        file_path: Path to the file to check. If None, returns None (pass-through).

    Returns:
        file_path: Absolute path to the validated file, or None if input was None

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if file_path is None:
        return None

    abs_path = os.path.abspath(file_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")

    return abs_path
