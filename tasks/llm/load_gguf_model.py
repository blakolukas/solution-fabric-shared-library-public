"""Load GGUF format LLM model using llama-cpp-python."""

import threading
from pathlib import Path
from typing import Any

from core.task import task


class ThreadSafeModel:
    """
    Thread-safe wrapper for LLM models.

    Automatically serializes inference calls using an internal lock,
    ensuring safe concurrent access when multiple tasks share the same model.
    This abstraction eliminates the need for users to manage locks manually.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._lock = threading.Lock()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute inference with automatic thread synchronization."""
        with self._lock:
            return self._model(*args, **kwargs)

    def __getattr__(self, name):
        """Proxy attribute access to the underlying model."""
        return getattr(self._model, name)

    def __repr__(self) -> str:
        return f"ThreadSafeModel({self._model!r})"


@task(
    outputs=["model"],
    display_name="Load LLM (GGUF)",
    description="Load a GGUF format LLM model using llama-cpp-python",
    category="llm",
    output_types={"model": "object"},
    parameters={
        "model_path": {
            "type": "str",
            "required": True,
            "description": "Path to the GGUF model file",
        },
        "n_ctx": {
            "type": "int",
            "required": False,
            "default": 4096,
            "description": "Context window size",
        },
        "n_threads": {
            "type": "int",
            "required": False,
            "default": None,
            "description": "Number of threads to use (None = auto)",
        },
        "n_gpu_layers": {
            "type": "int",
            "required": False,
            "default": -1,
            "description": "Number of layers to offload to GPU",
        },
        "verbose": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Enable verbose logging",
        },
    },
)
def load_gguf_model(
    model_path: str,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    verbose: bool = False,
) -> object:
    """
    Load a GGUF format LLM model.

    Args:
        model_path: Path to the GGUF model file
        n_ctx: Context window size
        n_gpu_layers: Number of layers to offload to GPU (-1 = all)
        verbose: Enable verbose logging

    Returns:
        Thread-safe model instance that automatically handles concurrent access

    Raises:
        FileNotFoundError: If the model file does not exist
    """
    from llama_cpp import Llama

    # Validate model path exists (similar to YOLO pattern)
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}\n"
            f"Please download a GGUF model and place it at the specified path.\n"
            f"You can download models from: https://huggingface.co/TheBloke"
        )

    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )

    return ThreadSafeModel(model)
