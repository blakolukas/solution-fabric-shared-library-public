"""Load a LLaMA tokenizer from HuggingFace."""

from core.task import task


@task(
    outputs=["tokenizer"],
    display_name="Load LLaMA Tokenizer",
    description="Load a LLaMA tokenizer from HuggingFace with proper configuration",
    category="llm",
    output_types={"tokenizer": "object"},
    parameters={
        "model_name": {
            "type": "str",
            "required": False,
            "default": "meta-llama/Meta-Llama-3-8B-Instruct",
            "description": "HuggingFace model identifier",
        },
        "padding_side": {
            "type": "str",
            "required": False,
            "default": "right",
            "description": "Where to add padding tokens (left, right)",
        },
        "add_eos_token": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "Add EOS token to sequences",
        },
        "add_bos_token": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "Add BOS token to sequences",
        },
        "trust_remote_code": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "Trust remote code from HuggingFace",
        },
    },
)
def load_llama_tokenizer(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    padding_side: str = "right",
    add_eos_token: bool = True,
    add_bos_token: bool = True,
    trust_remote_code: bool = True,
) -> object:
    """
    Load a LLaMA tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model identifier
        padding_side: Where to add padding tokens ("left", "right")
        add_eos_token: Add EOS token to sequences
        add_bos_token: Add BOS token to sequences
        trust_remote_code: Trust remote code from HuggingFace

    Returns:
        Loaded AutoTokenizer instance configured for LLaMA

    Example:
        Common models:
        - "meta-llama/Meta-Llama-3-8B-Instruct"
        - "meta-llama/Meta-Llama-3-70B-Instruct"
        - "meta-llama/Llama-2-7b-chat-hf"
    """
    from transformers import AutoTokenizer

    print(f"Loading LLaMA tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    # Configure tokenizer
    tokenizer.padding_side = padding_side

    # Set pad token to eos token if not already set (LLaMA models need this)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configure special tokens
    tokenizer.add_eos_token = add_eos_token
    tokenizer.add_bos_token = add_bos_token

    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    return tokenizer
