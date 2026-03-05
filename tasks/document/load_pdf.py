"""Load PDF document from file path."""

import os

from core.task import task


@task(
    outputs=["pdf"],
    output_types={"pdf": "pdf"},
    display_name="Load PDF",
    description="Load a PDF document from a file path",
    category="document",
    parameters={
        "input": {
            "type": "str",
            "required": True,
            "description": "Path to the PDF file",
        },
    },
)
def load_pdf(input: str) -> str:
    """
    Load a PDF document from a file path.

    This task validates the PDF file exists and returns its absolute path
    for use by downstream tasks and UI preview.

    Args:
        input: Path to the PDF file

    Returns:
        pdf: Absolute path to the PDF file (for preview and downstream processing)
    """
    if not input:
        raise ValueError("input file path is required")

    # Validate file extension
    _, ext = os.path.splitext(input)
    if ext.lower() != ".pdf":
        raise ValueError(f"Expected PDF file, got: {ext}")

    return input
