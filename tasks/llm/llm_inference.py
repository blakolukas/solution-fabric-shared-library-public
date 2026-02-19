"""Run inference on a loaded LLM model."""

from typing import Any

from core.task import task


def _detect_chat_format_stop_sequences(prompt: str) -> list[str]:
    """
    Auto-detect chat format from prompt content and return appropriate stop sequences.

    This allows workflows to omit explicit stop sequences - they are derived
    automatically from the prompt format markers.
    """
    if "<|im_start|>" in prompt:
        # ChatML format (Mistral, OpenChat, etc.)
        return ["<|im_end|>", "<|im_start|>"]
    elif "<|system|>" in prompt or "<|user|>" in prompt or "<|assistant|>" in prompt:
        # Zephyr format
        return ["</s>", "<|user|>", "<|system|>"]
    elif "[INST]" in prompt or "<<SYS>>" in prompt:
        # Llama 2/3 format
        return ["[/INST]", "[INST]"]
    elif "<|end|>" in prompt or "<|user|>" in prompt:
        # Phi format
        return ["<|end|>", "<|user|>", "<|assistant|>"]
    else:
        # Unknown format - no auto stop sequences
        return []


@task(
    outputs=["response"],
    display_name="LLM Inference",
    description="Run inference on a loaded LLM model",
    category="llm",
    output_types={"response": "text"},
    parameters={
        "model": {
            "type": "object",
            "required": True,
            "description": "Loaded LLM model instance (thread-safe)",
        },
        "prompt": {
            "type": "str",
            "required": True,
            "multiline": True,
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
            "description": "Sampling temperature (0 = deterministic, higher = more random)",
        },
        "top_p": {
            "type": "float",
            "required": False,
            "default": 0.9,
            "description": "Nucleus sampling parameter",
        },
        "repeat_penalty": {
            "type": "float",
            "required": False,
            "default": 1.1,
            "description": "Penalty for repeating tokens (1.0 = no penalty, higher = less repetition)",
        },
        "top_k": {
            "type": "int",
            "required": False,
            "default": 40,
            "description": "Top-k sampling parameter",
        },
    },
)
def llm_inference(
    model: Any,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    top_k: int = 40,
) -> str:
    """
    Run inference on a loaded LLM model.

    This is a generic, atomic inference task. The prompt should be
    fully constructed before calling this task (use format_template
    and build_chat_prompt tasks for prompt construction).

    Thread safety is handled automatically by the model wrapper.

    Stop sequences are auto-detected from the prompt format:
    - ChatML: <|im_end|>, <|im_start|>
    - Zephyr: </s>, <|user|>, <|system|>
    - Llama: [/INST], [INST]
    - Phi: <|end|>, <|user|>, <|assistant|>

    Args:
        model: Loaded LLM model instance
        prompt: Fully constructed prompt string
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = deterministic, higher = more random)
        top_p: Nucleus sampling parameter
        repeat_penalty: Penalty for repeating tokens (1.0 = no penalty, higher = less repetition)
        top_k: Top-k sampling parameter

    Returns:
        Generated text response
    """
    if prompt is None or not prompt.strip():
        return ""

    # Auto-detect stop sequences from prompt format if not explicitly provided
    effective_stop = _detect_chat_format_stop_sequences(prompt)

    output = model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=effective_stop,
        repeat_penalty=repeat_penalty,
        top_k=top_k,
    )

    # Extract the generated text
    if isinstance(output, dict) and "choices" in output:
        return output["choices"][0]["text"].strip()
    elif isinstance(output, str):
        return output.strip()
    else:
        return str(output)
