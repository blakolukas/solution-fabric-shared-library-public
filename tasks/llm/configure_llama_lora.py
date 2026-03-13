from core.task import task


@task(
    outputs=["lora_config"],
    output_types={"lora_config": "object"},
    parameters={
        "r": {
            "type": "int",
            "required": False,
            "default": 16,
            "description": "Rank of the low-rank matrices",
        },
        "lora_alpha": {
            "type": "int",
            "required": False,
            "default": 32,
            "description": "Scaling factor for LoRA",
        },
        "lora_dropout": {
            "type": "float",
            "required": False,
            "default": 0.05,
            "description": "Dropout probability for LoRA layers",
        },
        "target_modules": {
            "type": "list",
            "required": False,
            "default": None,
            "description": "List of module names to apply LoRA to",
        },
        "bias": {
            "type": "str",
            "required": False,
            "default": "none",
            "description": "Bias training strategy",
        },
        "task_type": {
            "type": "str",
            "required": False,
            "default": "CAUSAL_LM",
            "description": "Type of task",
        },
    },
)
def configure_llama_lora(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
):
    """Configure LoRA (Low-Rank Adaptation) for LLaMA 3 fine-tuning.

    Args:
        r: Rank of the low-rank matrices (higher = more capacity)
        lora_alpha: Scaling factor for LoRA (typically 2*r)
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        bias: Bias training strategy ("none", "all", "lora_only")
        task_type: Type of task (CAUSAL_LM for language modeling)
    """
    from peft import LoraConfig, TaskType

    # Default target modules for LLaMA 3 architecture
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    # Map string task type to enum
    task_type_mapping = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
    }

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type_mapping.get(task_type, TaskType.CAUSAL_LM),
    )

    print(f"LoRA Config: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Target modules: {target_modules}")

    return lora_config
