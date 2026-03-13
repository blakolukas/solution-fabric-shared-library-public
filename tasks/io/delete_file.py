import os

from core.task import task


@task(
    outputs=["deleted"],
    output_types={"deleted": "bool"},
    display_name="Delete File",
    description="Delete a file from disk",
    category="io",
    parameters={
        "file_path": {
            "type": "str",
            "required": True,
            "description": "Path to the file to delete",
        },
    },
    is_collapsed=True,
)
def delete_file(file_path: str):
    """
    Delete a file from disk.

    Args:
        file_path: Path to the file to delete

    Returns:
        True if deletion succeeded
    """
    os.remove(file_path)
    return True
