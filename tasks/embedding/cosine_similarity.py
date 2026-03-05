"""Calculate cosine similarity between two embeddings."""

from core.task import task


@task(
    outputs=["similarity"],
    display_name="Cosine Similarity",
    description="Calculate cosine similarity between two embeddings",
    category="embedding",
    output_types={"similarity": "float"},
    is_collapsed=True,
    parameters={
        "embedding_a": {
            "type": "array",
            "required": True,
            "description": "First embedding vector",
        },
        "embedding_b": {
            "type": "array",
            "required": True,
            "description": "Second embedding vector",
        },
    },
)
def cosine_similarity(
    embedding_a: list,
    embedding_b: list,
) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding_a: First embedding vector
        embedding_b: Second embedding vector

    Returns:
        Cosine similarity score (0 to 1 for normalized vectors)
    """
    import numpy as np

    if not embedding_a or not embedding_b:
        return 0.0

    a = np.array(embedding_a)
    b = np.array(embedding_b)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))
