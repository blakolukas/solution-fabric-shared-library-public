"""Generate embedding vector for a single text."""

from core.task import task


@task(
    outputs=["embedding"],
    display_name="Embed Text",
    description="Generate embedding vector for a single text",
    category="embedding",
    output_types={"embedding": "array"},
    is_collapsed=True,
    parameters={
        "text": {
            "type": "str",
            "required": True,
            "description": "Text to embed",
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
            "description": "Normalize the embedding to unit length",
        },
    },
)
def embed_text(
    text: str,
    model: object,
    normalize: bool = True,
) -> list:
    """
    Generate embedding vector for a single text.

    Args:
        text: Text to embed
        model: SentenceTransformer model
        normalize: Normalize the embedding to unit length

    Returns:
        Embedding vector as a list
    """
    if text is None or not str(text).strip():
        return []

    embedding = model.encode(
        str(text),
        normalize_embeddings=normalize,
    )

    return embedding.tolist()
