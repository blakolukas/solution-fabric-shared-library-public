"""Query a ChromaDB collection for similar documents."""

from typing import Any, Dict, List, Optional

from core.task import task


@task(
    outputs=["documents", "metadatas", "distances", "ids"],
    display_name="Query Collection",
    description="Query a ChromaDB collection for similar documents",
    category="vectorstore",
    output_types={"documents": "list", "metadatas": "list", "distances": "list", "ids": "list"},
    is_collapsed=True,
    parameters={
        "collection": {
            "type": "object",
            "required": True,
            "description": "ChromaDB collection",
        },
        "query_embeddings": {
            "type": "list",
            "required": False,
            "default": None,
            "description": "Query embedding vectors",
        },
        "query_texts": {
            "type": "list",
            "required": False,
            "default": None,
            "description": "Query texts (alternative to embeddings)",
        },
        "n_results": {
            "type": "int",
            "required": False,
            "default": 5,
            "description": "Number of results to return",
        },
        "where": {
            "type": "dict",
            "required": False,
            "default": None,
            "description": "Optional filter conditions",
        },
        "include": {
            "type": "list",
            "required": False,
            "default": None,
            "description": "What to include in results (documents, metadatas, distances)",
        },
    },
)
def query_collection(
    collection: object,
    query_embeddings: Optional[List[list]] = None,
    query_texts: Optional[List[str]] = None,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
) -> tuple:
    """
    Query a ChromaDB collection for similar documents.

    Args:
        collection: ChromaDB collection
        query_embeddings: Query embedding vectors
        query_texts: Query texts (alternative to embeddings)
        n_results: Number of results to return
        where: Optional filter conditions
        include: What to include in results (documents, metadatas, distances)

    Returns:
        Tuple of (documents, metadatas, distances, ids)
    """
    if include is None:
        include = ["documents", "metadatas", "distances"]

    kwargs = {
        "n_results": n_results,
        "include": include,
    }

    if query_embeddings:
        kwargs["query_embeddings"] = query_embeddings
    elif query_texts:
        kwargs["query_texts"] = query_texts
    else:
        return [], [], [], []

    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    # Extract results (ChromaDB returns nested lists for batch queries)
    documents = results.get("documents", [[]])[0] if results.get("documents") else []
    metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []
    ids = results.get("ids", [[]])[0] if results.get("ids") else []

    return documents, metadatas, distances, ids
