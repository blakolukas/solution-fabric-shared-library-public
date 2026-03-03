"""Generate text with token usage tracking."""

from typing import Optional

from core.task import task


@task(
    outputs=["response", "tokens_used"],
    display_name="LLM Generate",
    description="Generate text with token usage tracking",
    category="llm",
    output_types={"response": "text", "tokens_used": "int"},
    is_collapsed=True,
    parameters={
        "model": {
            "type": "object",
            "required": True,
            "description": "Loaded Llama model instance",
        },
        "prompt": {
            "type": "str",
            "required": True,
            "description": "Fully constructed prompt string",
        },
        "max_tokens": {
            "type": "int",
            "required": False,
            "default": 256,
            "description": "Maximum tokens to generate",
        },
        "temperature": {
            "type": "float",
            "required": False,
            "default": 0.7,
            "description": "Sampling temperature",
        },
        "top_p": {
            "type": "float",
            "required": False,
            "default": 0.9,
            "description": "Nucleus sampling parameter",
        },
        "stop": {
            "type": "list",
            "required": False,
            "default": None,
            "description": "Stop sequences",
        },
    },
)
def llm_generate(
    model: object,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop: Optional[list] = None,
) -> tuple:
    """
    Generate text with token usage tracking.

    Similar to llm_inference but also returns token count.

    Args:
        model: Loaded Llama model instance
        prompt: Fully constructed prompt string
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        stop: Stop sequences

    Returns:
        Tuple of (generated text, tokens used)
    """
    if prompt is None or not prompt.strip():
        return "", 0

    output = model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop or [],
    )

    # Extract results
    if isinstance(output, dict):
        text = output.get("choices", [{}])[0].get("text", "").strip()
        usage = output.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        return text, tokens
    else:
        return str(output).strip(), 0
