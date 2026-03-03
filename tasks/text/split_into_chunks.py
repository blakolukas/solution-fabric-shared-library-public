"""Split text into overlapping chunks of specified size."""

from typing import List

from core.task import task


@task(
    outputs=["chunks"],
    display_name="Split Into Chunks",
    description="Split text into overlapping chunks of specified size",
    category="text",
    output_types={"chunks": "list"},
    is_collapsed=True,
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to split",
        },
        "chunk_size": {
            "type": "int",
            "required": False,
            "default": 1000,
            "description": "Target size of each chunk in characters",
        },
        "chunk_overlap": {
            "type": "int",
            "required": False,
            "default": 200,
            "description": "Overlap between consecutive chunks",
        },
        "separator": {
            "type": "str",
            "required": False,
            "default": "\n\n",
            "description": "Preferred split point",
        },
    },
)
def split_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separator: str = "\n\n",
) -> List[str]:
    """
    Split text into overlapping chunks.

    This is essential for RAG workflows where long documents need to be
    split into manageable pieces for embedding and retrieval.

    Args:
        text: Text to split
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks
        separator: Preferred split point (tries to split here first)

    Returns:
        List of text chunks
    """
    if text is None or not text.strip():
        return []

    text = str(text)

    # If text is smaller than chunk size, return as single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # Calculate end position
        end = start + chunk_size

        if end >= len(text):
            # Last chunk
            chunks.append(text[start:])
            break

        # Try to find a good split point
        chunk = text[start:end]

        # Look for separator within the chunk
        split_pos = chunk.rfind(separator)
        if split_pos == -1:
            # No separator found, try newline
            split_pos = chunk.rfind("\n")
        if split_pos == -1:
            # No newline, try space
            split_pos = chunk.rfind(" ")
        if split_pos == -1:
            # No good split point, use chunk_size
            split_pos = chunk_size

        # Add the chunk
        chunks.append(text[start : start + split_pos + len(separator)])

        # Move start position (account for overlap)
        start = start + split_pos + len(separator) - chunk_overlap

        # Ensure we make progress
        if start <= 0:
            start = split_pos + len(separator)

    return chunks
