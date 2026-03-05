"""Add documents with embeddings to a ChromaDB collection."""

from typing import Any, Dict, List, Optional

from core.task import task


@task(
    outputs=["success", "ids"],
    display_name="Add Documents",
    description="Add documents with embeddings to a ChromaDB collection",
    category="vectorstore",
    output_types={"success": "bool", "ids": "list"},
    is_collapsed=True,
    parameters={
        "collection": {
            "type": "object",
            "required": True,
            "description": "ChromaDB collection",
        },
        "documents": {
            "type": "list",
            "required": True,
            "description": "List of document texts",
        },
        "embeddings": {
            "type": "list",
            "required": False,
            "default": None,
            "description": "Pre-computed embeddings (optional if collection has embedding fn)",
        },
        "metadatas": {
            "type": "list",
            "required": False,
            "default": None,
            "description": "Optional metadata for each document",
        },
        "ids": {
            "type": "list",
            "required": False,
            "default": None,
            "description": "Optional IDs (auto-generated if not provided)",
        },
    },
)
def add_documents(
    collection: object,
    documents: List[str],
    embeddings: Optional[List[list]] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
) -> tuple:
    """
    Add documents to a ChromaDB collection.

    Args:
        collection: ChromaDB collection
        documents: List of document texts
        embeddings: Pre-computed embeddings (optional if collection has embedding fn)
        metadatas: Optional metadata for each document
        ids: Optional IDs (auto-generated if not provided)

    Returns:
        Tuple of (success, list of IDs)
    """
    import uuid

    if not documents:
        return True, []

    # Generate IDs if not provided
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in documents]

    kwargs = {
        "documents": documents,
        "ids": ids,
    }

    if embeddings:
        kwargs["embeddings"] = embeddings
    if metadatas:
        kwargs["metadatas"] = metadatas

    collection.add(**kwargs)

    return True, ids
