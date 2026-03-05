"""Run inference on a loaded LLM model"""

from typing import Any, Generator

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
    description="Run inference on a loaded LLM model.",
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
        "streaming": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Emit tokens one-by-one as they are generated (token streaming). When True the output is a generator; the workflow engine streams each token to the UI in real-time.",
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
    streaming: bool = False,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    top_k: int = 40,
):
    """
    Run inference on a loaded LLM model.

    This is a generic, atomic inference task. The prompt should be
    fully constructed before calling this task (use format_template
    and build_chat_prompt tasks for prompt construction).

    Thread safety is handled automatically by the model wrapper.

    When ``streaming=True`` the function returns a generator that yields tokens
    one-by-one. The workflow engine detects the generator at runtime, enters its
    reactive streaming mode, and sends each token to the UI as a ``token_chunk``
    WebSocket message. The caller accumulates the tokens; the final joined string
    is the ``response`` output at execution completion.

    When ``streaming=False`` (default) the full response string is returned
    synchronously after generation finishes.

    Stop sequences are auto-detected from the prompt format:
    - ChatML: <|im_end|>, <|im_start|>
    - Zephyr: </s>, <|user|>, <|system|>
    - Llama: [/INST], [INST]
    - Phi: <|end|>, <|user|>, <|assistant|>

    Args:
        model: Loaded LLM model instance
        prompt: Fully constructed prompt string
        streaming: If True, yield tokens one-by-one (token streaming mode)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = deterministic, higher = more random)
        top_p: Nucleus sampling parameter
        repeat_penalty: Penalty for repeating tokens (1.0 = no penalty, higher = less repetition)
        top_k: Top-k sampling parameter

    Returns:
        str when streaming=False; Generator[str, None, None] when streaming=True
    """
    if prompt is None or not prompt.strip():
        return "" if not streaming else _empty_generator()

    effective_stop = _detect_chat_format_stop_sequences(prompt)

    if streaming:
        # Return a generator so the workflow engine detects it as streamable and
        # enters reactive streaming mode, sending each token to the UI in real-time.
        return _token_stream_generator(
            model, prompt, max_tokens, temperature, top_p, effective_stop, repeat_penalty, top_k
        )

    # --- Non-streaming path: return the full response string ---
    output = model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=effective_stop,
        repeat_penalty=repeat_penalty,
        top_k=top_k,
    )

    if isinstance(output, dict) and "choices" in output:
        return output["choices"][0]["text"].strip()
    elif isinstance(output, str):
        return output.strip()
    else:
        return str(output)


def _empty_generator() -> Generator[str, None, None]:
    """Yield a single empty string for the streaming=True + empty-prompt edge case."""
    yield ""


def _token_stream_generator(
    model: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list,
    repeat_penalty: float,
    top_k: int,
) -> Generator[str, None, None]:
    """
    Inner generator that yields tokens from model.stream().

    Kept as a standalone function (not nested inside llm_inference) so that
    llm_inference itself is a regular function, not a generator function.
    Python requires a function to be *either* a regular function (uses return)
    *or* a generator function (uses yield) — this separation satisfies that
    constraint while still returning a generator object when streaming=True.
    """
    token_stream = model.stream(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        repeat_penalty=repeat_penalty,
        top_k=top_k,
    )
    for chunk in token_stream:
        if isinstance(chunk, dict) and "choices" in chunk:
            token_text = chunk["choices"][0].get("text", "")
        elif isinstance(chunk, str):
            token_text = chunk
        else:
            token_text = str(chunk)
        if token_text:
            yield token_text

