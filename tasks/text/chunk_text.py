"""Split text into chunks with metadata."""

from core.task import task


@task(
    outputs=["chunks", "count"],
    display_name="Chunk Text",
    description="Split text into chunks with metadata",
    category="text",
    output_types={"chunks": "list", "count": "int"},
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
            "default": 500,
            "description": "Target size of each chunk",
        },
        "chunk_overlap": {
            "type": "int",
            "required": False,
            "default": 50,
            "description": "Overlap between chunks",
        },
    },
)
def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> tuple:
    """
    Split text into chunks and return count.

    A simpler version of split_into_chunks that also returns the chunk count.

    Args:
        text: Text to split
        chunk_size: Target size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        Tuple of (chunks list, chunk count)
    """

    if text is None or not text.strip():
        return [], 0

    text = str(text)

    # If text is smaller than chunk size, return as single chunk
    if len(text) <= chunk_size:
        return [text], 1

    chunks = []
    start = 0
    separator = "\n\n"

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

    return chunks, len(chunks)
