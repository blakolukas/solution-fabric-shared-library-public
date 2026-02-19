"""Get or create a ChromaDB collection."""

from typing import Optional

from core.task import task


@task(
    outputs=["collection"],
    display_name="Get Or Create Collection",
    description="Get or create a ChromaDB collection",
    category="vectorstore",
    output_types={"collection": "object"},
    parameters={
        "client": {
            "type": "object",
            "required": True,
            "description": "ChromaDB client",
        },
        "collection_name": {
            "type": "str",
            "required": False,
            "default": "default",
            "description": "Name of the collection",
        },
        "embedding_function": {
            "type": "object",
            "required": False,
            "default": None,
            "description": "Optional custom embedding function",
        },
    },
)
def get_or_create_collection(
    client: object,
    collection_name: str = "default",
    embedding_function: Optional[object] = None,
) -> object:
    """
    Get or create a ChromaDB collection.

    Args:
        client: ChromaDB client
        collection_name: Name of the collection
        embedding_function: Optional custom embedding function

    Returns:
        ChromaDB collection
    """
    kwargs = {"name": collection_name}
    if embedding_function:
        kwargs["embedding_function"] = embedding_function

    return client.get_or_create_collection(**kwargs)
