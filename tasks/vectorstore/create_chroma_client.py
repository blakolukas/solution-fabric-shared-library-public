"""Create a ChromaDB client."""

from typing import Optional

from core.task import task


@task(
    outputs=["client"],
    display_name="Create Chroma Client",
    description="Create a ChromaDB client (ephemeral or persistent)",
    category="vectorstore",
    output_types={"client": "object"},
    parameters={
        "persist_directory": {
            "type": "str",
            "required": False,
            "default": None,
            "description": "Path to persist data (None = ephemeral/in-memory)",
        },
    },
)
def create_chroma_client(
    persist_directory: Optional[str] = None,
) -> object:
    """
    Create a ChromaDB client.

    Args:
        persist_directory: Path to persist data (None = ephemeral/in-memory)

    Returns:
        ChromaDB client instance
    """
    import chromadb

    if persist_directory:
        client = chromadb.PersistentClient(path=persist_directory)
    else:
        client = chromadb.Client()

    return client
