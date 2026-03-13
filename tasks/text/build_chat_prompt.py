"""Build a chat-style prompt with system and user messages."""

from core.task import task


@task(
    outputs=["prompt"],
    display_name="Build Chat Prompt",
    description="Build a chat-style prompt with system and user messages",
    category="text",
    output_types={"prompt": "text"},
    parameters={
        "user_message": {
            "type": "str",
            "required": True,
            "description": "The user's message/question",
        },
        "system_message": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "Optional system instructions",
        },
        "assistant_prefix": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "Optional prefix for assistant response",
        },
        "format_style": {
            "type": "str",
            "required": False,
            "default": "llama",
            "options": ["llama", "chatml", "zephyr", "simple", "raw"],
            "description": "Prompt format style",
        },
    },
)
def build_chat_prompt(
    user_message: str,
    system_message: str = "",
    assistant_prefix: str = "",
    format_style: str = "llama",
) -> str:
    """
    Build a chat-style prompt with system and user messages.

    Args:
        user_message: The user's message/question
        system_message: Optional system instructions
        assistant_prefix: Optional prefix for assistant response
        format_style: Prompt format style:
            - "llama": Llama 2/3 style with [INST] tags
            - "chatml": ChatML style with <|im_start|> tags
            - "simple": Simple "System: ... User: ... Assistant:" format
            - "raw": Just concatenate messages with newlines

    Returns:
        Formatted prompt string
    """
    if format_style == "llama":
        parts = []
        if system_message:
            parts.append(f"<<SYS>>\n{system_message}\n<</SYS>>\n\n")
        parts.append(f"[INST] {user_message} [/INST]")
        if assistant_prefix:
            parts.append(f" {assistant_prefix}")
        return "".join(parts)

    elif format_style == "chatml":
        parts = []
        if system_message:
            parts.append(f"<|im_start|>system\n{system_message}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user_message}<|im_end|>")
        parts.append("<|im_start|>assistant")
        if assistant_prefix:
            parts.append(f"\n{assistant_prefix}")
        return "\n".join(parts)

    elif format_style == "zephyr":
        parts = []
        if system_message:
            parts.append(f"<|system|>\n{system_message}</s>")
        parts.append(f"<|user|>\n{user_message}</s>")
        parts.append("<|assistant|>")
        if assistant_prefix:
            parts.append(f"\n{assistant_prefix}")
        return "\n".join(parts)

    elif format_style == "simple":
        parts = []
        if system_message:
            parts.append(f"System: {system_message}")
        parts.append(f"User: {user_message}")
        parts.append("Assistant:")
        if assistant_prefix:
            parts[-1] += f" {assistant_prefix}"
        return "\n\n".join(parts)

    else:  # raw
        parts = []
        if system_message:
            parts.append(system_message)
        parts.append(user_message)
        if assistant_prefix:
            parts.append(assistant_prefix)
        return "\n\n".join(parts)
