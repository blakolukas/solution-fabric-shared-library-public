"""Load a LLaMA model from HuggingFace."""

from typing import Optional

from core.task import task


@task(
    outputs=["model"],
    display_name="Load LLaMA Model",
    description="Load a LLaMA model from HuggingFace with optional quantization",
    category="llm",
    output_types={"model": "object"},
    parameters={
        "model_name": {
            "type": "str",
            "required": False,
            "default": "meta-llama/Meta-Llama-3-8B-Instruct",
            "description": "HuggingFace model identifier",
        },
        "device_map": {
            "type": "str",
            "required": False,
            "default": "auto",
            "description": "Device placement strategy (auto, cuda, cpu)",
        },
        "torch_dtype": {
            "type": "str",
            "required": False,
            "default": "bfloat16",
            "description": "Data type for model weights (float32, float16, bfloat16)",
        },
        "load_in_8bit": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Load model in 8-bit quantization",
        },
        "load_in_4bit": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Load model in 4-bit quantization",
        },
        "attn_implementation": {
            "type": "str",
            "required": False,
            "default": None,
            "description": "Attention implementation (None, flash_attention_2, sdpa)",
        },
        "trust_remote_code": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "Trust remote code from HuggingFace",
        },
    },
)
def load_llama_model(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    attn_implementation: Optional[str] = None,
    trust_remote_code: bool = True,
) -> object:
    """
    Load a LLaMA model from HuggingFace with optional quantization.

    Args:
        model_name: HuggingFace model identifier
        device_map: Device placement strategy ("auto", "cuda", "cpu")
        torch_dtype: Data type for model weights ("float32", "float16", "bfloat16")
        load_in_8bit: Load model in 8-bit quantization (requires bitsandbytes)
        load_in_4bit: Load model in 4-bit quantization (requires bitsandbytes)
        attn_implementation: Attention implementation (None, "flash_attention_2", "sdpa")
        trust_remote_code: Trust remote code from HuggingFace

    Returns:
        Loaded AutoModelForCausalLM instance

    Example:
        Common models:
        - "meta-llama/Meta-Llama-3-8B-Instruct"
        - "meta-llama/Meta-Llama-3-70B-Instruct"
        - "meta-llama/Llama-2-7b-chat-hf"

    Notes:
        - 4-bit quantization is recommended for large models on consumer GPUs
        - flash_attention_2 requires flash-attn package and compatible GPU
        - bfloat16 is recommended for training/fine-tuning
    """
    import torch
    from transformers import AutoModelForCausalLM

    # Map string dtype to torch dtype
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    kwargs = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype_mapping.get(torch_dtype, torch.bfloat16),
    }

    # Add attention implementation if specified
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    # Configure quantization (mutually exclusive)
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
        print(f"Loading {model_name} with 8-bit quantization...")
    elif load_in_4bit:
        kwargs["load_in_4bit"] = True
        print(f"Loading {model_name} with 4-bit quantization...")
    else:
        print(f"Loading {model_name} in {torch_dtype} precision...")

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    print(f"Model loaded successfully on {device_map}")
    return model
