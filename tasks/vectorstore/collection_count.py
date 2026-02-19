"""Get the number of documents in a collection."""

from core.task import task


@task(
    outputs=["count"],
    display_name="Collection Count",
    description="Get the number of documents in a collection",
    category="vectorstore",
    output_types={"count": "int"},
    parameters={
        "collection": {
            "type": "object",
            "required": True,
            "description": "ChromaDB collection",
        },
    },
)
def collection_count(collection: object) -> int:
    """
    Get the number of documents in a collection.

    Args:
        collection: ChromaDB collection

    Returns:
        Number of documents
    """
    return collection.count()
