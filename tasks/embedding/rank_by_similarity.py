"""Rank embeddings by similarity to a query embedding."""

from typing import List

from core.task import task


@task(
    outputs=["scores", "indices"],
    display_name="Rank By Similarity",
    description="Rank embeddings by similarity to a query embedding",
    category="embedding",
    output_types={"scores": "list", "indices": "list"},
    is_collapsed=True,
    parameters={
        "query_embedding": {
            "type": "array",
            "required": True,
            "description": "Query embedding vector",
        },
        "embeddings": {
            "type": "list",
            "required": True,
            "description": "List of embeddings to rank",
        },
        "top_k": {
            "type": "int",
            "required": False,
            "default": 5,
            "description": "Number of top results to return",
        },
    },
)
def rank_by_similarity(
    query_embedding: list,
    embeddings: List[list],
    top_k: int = 5,
) -> tuple:
    """
    Rank embeddings by similarity to a query embedding.

    Args:
        query_embedding: Query embedding vector
        embeddings: List of embeddings to rank
        top_k: Number of top results to return

    Returns:
        Tuple of (similarity scores, indices of top results)
    """
    import numpy as np

    if not query_embedding or not embeddings:
        return [], []

    query = np.array(query_embedding)
    embs = np.array(embeddings)

    # Calculate cosine similarities
    similarities = np.dot(embs, query) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(query) + 1e-8)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return (
        [float(similarities[i]) for i in top_indices],
        [int(i) for i in top_indices],
    )
