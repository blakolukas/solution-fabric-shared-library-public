"""Build a messages list for chat-style LLM APIs."""

from typing import Optional

from core.task import task


@task(
    outputs=["messages"],
    display_name="Build Messages",
    description="Build a messages list for chat-style LLM APIs",
    category="llm",
    output_types={"messages": "list"},
    is_collapsed=True,
    parameters={
        "user_message": {
            "type": "str",
            "required": True,
            "description": "The user's message",
        },
        "system_message": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "Optional system instructions",
        },
        "history": {
            "type": "list",
            "required": False,
            "default": None,
            "description": "Optional list of previous messages",
        },
    },
)
def build_messages(
    user_message: str,
    system_message: str = "",
    history: Optional[list] = None,
) -> list:
    """
    Build a messages list for chat-style APIs.

    Creates a list of message dicts in the format expected by
    chat completion APIs.

    Args:
        user_message: The user's message
        system_message: Optional system instructions
        history: Optional list of previous messages

    Returns:
        List of message dictionaries
    """
    messages = []

    if system_message:
        messages.append({"role": "system", "content": system_message})

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_message})

    return messages
