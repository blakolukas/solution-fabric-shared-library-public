"""Tokenize a text prompt for LLM input."""

from core.task import task


@task(
    outputs=["tokenized"],
    display_name="Tokenize Prompt",
    description="Tokenize a text prompt for LLM input using HuggingFace tokenizer",
    category="llm",
    output_types={"tokenized": "object"},
    parameters={
        "prompt": {
            "type": "str",
            "required": True,
            "description": "The text prompt to tokenize",
        },
        "tokenizer": {
            "type": "object",
            "required": True,
            "description": "HuggingFace tokenizer instance",
        },
        "max_length": {
            "type": "int",
            "required": False,
            "default": 2048,
            "description": "Maximum sequence length",
        },
        "padding": {
            "type": "str",
            "required": False,
            "default": "max_length",
            "description": "Padding strategy (max_length, longest, do_not_pad)",
        },
        "truncation": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "Enable truncation to max_length",
        },
        "return_tensors": {
            "type": "str",
            "required": False,
            "default": "pt",
            "description": "Type of tensors to return (pt, tf, np, None)",
        },
    },
)
def tokenize_prompt(
    prompt: str,
    tokenizer: object,
    max_length: int = 2048,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str = "pt",
) -> dict:
    """
    Tokenize a text prompt for LLM input.

    Args:
        prompt: The text prompt to tokenize
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length (default: 2048)
        padding: Padding strategy - "max_length", "longest", or "do_not_pad" (default: "max_length")
        truncation: Enable truncation to max_length (default: True)
        return_tensors: Type of tensors to return - "pt" (PyTorch), "tf" (TensorFlow), "np" (NumPy), or None (default: "pt")

    Returns:
        Dictionary containing tokenized inputs with keys:
        - input_ids: Token IDs as tensors
        - attention_mask: Attention mask as tensors
        (Additional keys may be present depending on tokenizer)

    Example:
        For single prompt inference, use return_tensors="pt"
        For batch dataset preparation, use return_tensors=None
    """
    inputs = tokenizer(
        prompt,
        return_tensors=return_tensors,
        truncation=truncation,
        max_length=max_length,
        padding=padding,
    )
    return inputs
