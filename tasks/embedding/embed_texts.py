"""Generate embedding vectors for multiple texts."""

from typing import List

from core.task import task


@task(
    outputs=["embeddings"],
    display_name="Embed Texts",
    description="Generate embedding vectors for multiple texts",
    category="embedding",
    output_types={"embeddings": "list"},
    parameters={
        "texts": {
            "type": "list",
            "required": True,
            "description": "List of texts to embed",
        },
        "model": {
            "type": "object",
            "required": True,
            "description": "SentenceTransformer model",
        },
        "normalize": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "Normalize embeddings to unit length",
        },
        "batch_size": {
            "type": "int",
            "required": False,
            "default": 32,
            "description": "Batch size for encoding",
        },
        "show_progress": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Show progress bar",
        },
    },
)
def embed_texts(
    texts: List[str],
    model: object,
    normalize: bool = True,
    batch_size: int = 32,
    show_progress: bool = False,
) -> List[list]:
    """
    Generate embedding vectors for multiple texts.

    Args:
        texts: List of texts to embed
        model: SentenceTransformer model
        normalize: Normalize embeddings to unit length
        batch_size: Batch size for encoding
        show_progress: Show progress bar

    Returns:
        List of embedding vectors
    """
    if texts is None or len(texts) == 0:
        return []

    # Filter out None/empty texts
    valid_texts = [str(t) for t in texts if t is not None and str(t).strip()]

    if not valid_texts:
        return []

    embeddings = model.encode(
        valid_texts,
        normalize_embeddings=normalize,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )

    return [e.tolist() for e in embeddings]
