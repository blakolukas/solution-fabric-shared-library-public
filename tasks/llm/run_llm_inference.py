"""Run inference on a HuggingFace LLM model."""

from core.task import task


@task(
    outputs=["generated_ids"],
    display_name="Run LLM Inference",
    description="Run inference on a HuggingFace LLM model with tokenized input",
    category="llm",
    output_types={"generated_ids": "object"},
    parameters={
        "tokenized": {
            "type": "object",
            "required": True,
            "description": "Tokenized input dictionary with input_ids and attention_mask",
        },
        "model": {
            "type": "object",
            "required": True,
            "description": "HuggingFace model instance",
        },
        "tokenizer": {
            "type": "object",
            "required": True,
            "description": "HuggingFace tokenizer instance",
        },
        "max_new_tokens": {
            "type": "int",
            "required": False,
            "default": 50,
            "description": "Maximum tokens to generate",
        },
        "min_new_tokens": {
            "type": "int",
            "required": False,
            "default": 10,
            "description": "Minimum tokens to generate",
        },
        "temperature": {
            "type": "float",
            "required": False,
            "default": 0.3,
            "description": "Sampling temperature (0.0 = greedy, higher = more random)",
        },
        "top_p": {
            "type": "float",
            "required": False,
            "default": 0.9,
            "description": "Top-p (nucleus) sampling parameter",
        },
        "do_sample": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "Enable sampling (False = greedy decoding)",
        },
    },
)
def run_llm_inference(
    tokenized: dict,
    model: object,
    tokenizer: object,
    max_new_tokens: int = 50,
    min_new_tokens: int = 10,
    temperature: float = 0.3,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> object:
    """
    Run inference on a HuggingFace LLM model with tokenized input.

    Args:
        tokenized: Tokenized input dictionary (from tokenize_prompt task)
        model: HuggingFace model instance (from load_llama_model task)
        tokenizer: HuggingFace tokenizer instance (for pad_token_id)
        max_new_tokens: Maximum tokens to generate (default: 50)
        min_new_tokens: Minimum tokens to generate (default: 10)
        temperature: Sampling temperature - 0.0 for deterministic, higher for creative (default: 0.3)
        top_p: Top-p (nucleus) sampling parameter (default: 0.9)
        do_sample: Enable sampling; False uses greedy decoding (default: True)

    Returns:
        Generated output token IDs (tensor)

    Notes:
        - Temperature controls randomness: 0.1-0.3 for factual, 0.7-1.0 for creative
        - Use decode_llm_output task to convert generated_ids to text
        - Input is automatically moved to model's device
    """
    import torch

    # Move input tensors to model's device
    if hasattr(model, "device"):
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}

    # Run inference without gradient computation
    with torch.no_grad():
        outputs = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    return outputs
