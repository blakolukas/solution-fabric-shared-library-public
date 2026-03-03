"""Retrieve similar documents using embedding query."""

from typing import Optional

from core.task import task


@task(
    outputs=["documents", "scores"],
    display_name="Retrieve Similar",
    description="Retrieve similar documents using embedding query",
    category="vectorstore",
    output_types={"documents": "list", "scores": "list"},
    is_collapsed=True,
    parameters={
        "collection": {
            "type": "object",
            "required": True,
            "description": "ChromaDB collection",
        },
        "query_embedding": {
            "type": "array",
            "required": True,
            "description": "Query embedding vector",
        },
        "top_k": {
            "type": "int",
            "required": False,
            "default": 5,
            "description": "Number of results to retrieve",
        },
        "score_threshold": {
            "type": "float",
            "required": False,
            "default": None,
            "description": "Minimum similarity score (filters out low relevance)",
        },
    },
)
def retrieve_similar(
    collection: object,
    query_embedding: list,
    top_k: int = 5,
    score_threshold: Optional[float] = None,
) -> tuple:
    """
    Retrieve similar documents from a collection.

    Simplified retrieval interface for common RAG use case.

    Args:
        collection: ChromaDB collection
        query_embedding: Query embedding vector
        top_k: Number of results to retrieve
        score_threshold: Minimum similarity score (filters out low relevance)

    Returns:
        Tuple of (document texts, similarity scores)
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"],
    )

    documents = results.get("documents", [[]])[0] if results.get("documents") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    # Convert distances to similarity scores (ChromaDB uses L2 distance by default)
    # Lower distance = higher similarity
    scores = [1 / (1 + d) for d in distances]

    # Apply score threshold filter
    if score_threshold is not None:
        filtered = [(doc, score) for doc, score in zip(documents, scores) if score >= score_threshold]
        if filtered:
            documents, scores = zip(*filtered)
            documents, scores = list(documents), list(scores)
        else:
            documents, scores = [], []

    return documents, scores
