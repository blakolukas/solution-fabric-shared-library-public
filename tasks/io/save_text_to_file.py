import os

from core.task import task


@task(
    outputs=["file_path"],
    display_name="Save Text to File",
    description="Save text content to a file",
    category="io",
    output_types={"file_path": "text"},
    is_collapsed=True,
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text content to save",
        },
        "directory": {
            "type": "str",
            "required": True,
            "description": "Directory to save the file",
        },
        "filename": {
            "type": "str",
            "required": True,
            "description": "Filename (e.g., 'output.txt')",
        },
        "encoding": {
            "type": "str",
            "required": False,
            "default": "utf-8",
            "description": "Text encoding (default: utf-8)",
        },
    },
)
def save_text_to_file(text: str, directory: str, filename: str, encoding: str = "utf-8"):
    """
    Save text content to a file.

    Args:
        text: Text content to save
        directory: Directory to save the file
        filename: Filename (e.g., 'output.txt')
        encoding: Text encoding (default: utf-8)

    Returns:
        Full path to the saved file
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)

    with open(file_path, "w", encoding=encoding) as f:
        f.write(text)

    return file_path
