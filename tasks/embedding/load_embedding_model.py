"""Load a sentence transformer embedding model."""

from core.task import task


@task(
    outputs=["model"],
    display_name="Load Embedding Model",
    description="Load a sentence transformer embedding model",
    category="embedding",
    output_types={"model": "object"},
    parameters={
        "model_name": {
            "type": "str",
            "required": False,
            "default": "all-MiniLM-L6-v2",
            "description": "Name of the model from HuggingFace",
        },
        "device": {
            "type": "str",
            "required": False,
            "default": "cpu",
            "description": "Device to run on (cpu, cuda, mps)",
        },
    },
)
def load_embedding_model(
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
) -> object:
    """
    Load a sentence transformer embedding model.

    Args:
        model_name: Name of the model from HuggingFace
                   Common options:
                   - "all-MiniLM-L6-v2" (fast, good quality)
                   - "all-mpnet-base-v2" (better quality, slower)
                   - "multi-qa-MiniLM-L6-cos-v1" (optimized for QA)
        device: Device to run on ("cpu", "cuda", "mps")

    Returns:
        Loaded SentenceTransformer model
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return model
