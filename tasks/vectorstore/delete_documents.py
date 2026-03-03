"""Delete documents from a collection by IDs."""

from typing import List

from core.task import task


@task(
    outputs=["success"],
    display_name="Delete Documents",
    description="Delete documents from a collection by IDs",
    category="vectorstore",
    output_types={"success": "bool"},
    is_collapsed=True,
    parameters={
        "collection": {
            "type": "object",
            "required": True,
            "description": "ChromaDB collection",
        },
        "ids": {
            "type": "list",
            "required": True,
            "description": "List of document IDs to delete",
        },
    },
)
def delete_documents(
    collection: object,
    ids: List[str],
) -> bool:
    """
    Delete documents from a collection.

    Args:
        collection: ChromaDB collection
        ids: List of document IDs to delete

    Returns:
        True if successful
    """
    if not ids:
        return True

    collection.delete(ids=ids)
    return True
