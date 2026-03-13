"""Run chat completion on a model with messages."""

from core.task import task


@task(
    outputs=["response"],
    display_name="Chat Completion",
    description="Run chat completion on a model with messages",
    category="llm",
    output_types={"response": "text"},
    parameters={
        "model": {
            "type": "object",
            "required": True,
            "description": "Loaded Llama model instance",
        },
        "messages": {
            "type": "list",
            "required": True,
            "description": "List of message dictionaries",
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
    },
)
def chat_completion(
    model: object,
    messages: list,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """
    Run chat completion using a messages list.

    Uses the model's create_chat_completion method if available,
    otherwise falls back to formatting messages as a prompt.

    Args:
        model: Loaded Llama model instance
        messages: List of message dictionaries
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated response text
    """
    if not messages:
        return ""

    # Try chat completion if available
    if hasattr(model, "create_chat_completion"):
        output = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if isinstance(output, dict) and "choices" in output:
            return output["choices"][0]["message"]["content"].strip()
        return str(output)

    # Fallback: format messages as prompt
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt_parts.append(f"{role.capitalize()}: {content}")
    prompt_parts.append("Assistant:")

    prompt = "\n\n".join(prompt_parts)

    output = model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["User:", "Human:"],
    )

    if isinstance(output, dict) and "choices" in output:
        return output["choices"][0]["text"].strip()
    return str(output).strip()
