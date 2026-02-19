from core.task import task


@task(
    outputs=["training_args"],
    parameters={
        "output_dir": {
            "type": "str",
            "required": False,
            "default": "./llama_finetuned",
            "description": "Directory to save model checkpoints",
        },
        "num_train_epochs": {
            "type": "int",
            "required": False,
            "default": 3,
            "description": "Number of training epochs",
        },
        "per_device_train_batch_size": {
            "type": "int",
            "required": False,
            "default": 2,
            "description": "Batch size per device for training",
        },
        "per_device_eval_batch_size": {
            "type": "int",
            "required": False,
            "default": 2,
            "description": "Batch size per device for evaluation",
        },
        "gradient_accumulation_steps": {
            "type": "int",
            "required": False,
            "default": 8,
            "description": "Number of gradient accumulation steps",
        },
        "learning_rate": {
            "type": "float",
            "required": False,
            "default": 2e-4,
            "description": "Initial learning rate",
        },
        "weight_decay": {
            "type": "float",
            "required": False,
            "default": 0.01,
            "description": "Weight decay coefficient",
        },
        "warmup_ratio": {
            "type": "float",
            "required": False,
            "default": 0.03,
            "description": "Ratio of warmup steps",
        },
        "lr_scheduler_type": {
            "type": "str",
            "required": False,
            "default": "cosine",
            "description": "Learning rate scheduler type",
        },
        "logging_steps": {
            "type": "int",
            "required": False,
            "default": 10,
            "description": "Log every N steps",
        },
        "save_steps": {
            "type": "int",
            "required": False,
            "default": 500,
            "description": "Save checkpoint every N steps",
        },
        "save_total_limit": {
            "type": "int",
            "required": False,
            "default": 2,
            "description": "Maximum number of checkpoints to keep",
        },
        "fp16": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Use mixed precision training (FP16)",
        },
        "bf16": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "Use bfloat16 precision training",
        },
        "gradient_checkpointing": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "Enable gradient checkpointing",
        },
        "optim": {
            "type": "str",
            "required": False,
            "default": "paged_adamw_32bit",
            "description": "Optimizer to use",
        },
        "max_grad_norm": {
            "type": "float",
            "required": False,
            "default": 0.3,
            "description": "Maximum gradient norm for clipping",
        },
        "group_by_length": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "Group sequences by length",
        },
        "report_to": {
            "type": "str",
            "required": False,
            "default": "none",
            "description": "Reporting destination",
        },
        "remove_unused_columns": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Remove unused columns from dataset",
        },
    },
)
def configure_llama_training_arguments(
    output_dir: str = "./llama_finetuned",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    lr_scheduler_type: str = "cosine",
    logging_steps: int = 10,
    save_steps: int = 500,
    save_total_limit: int = 2,
    fp16: bool = False,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    optim: str = "paged_adamw_32bit",
    max_grad_norm: float = 0.3,
    group_by_length: bool = True,
    report_to: str = "none",
    remove_unused_columns: bool = False,
):
    """Configure training arguments for LLaMA 3 fine-tuning.

    Args:
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device for training
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Initial learning rate (higher for LoRA)
        weight_decay: Weight decay coefficient
        warmup_ratio: Ratio of warmup steps
        lr_scheduler_type: Learning rate scheduler type
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        save_total_limit: Maximum number of checkpoints to keep
        fp16: Use mixed precision training (FP16)
        bf16: Use bfloat16 precision training (recommended for LLaMA)
        gradient_checkpointing: Enable gradient checkpointing to save memory
        optim: Optimizer to use (paged_adamw for memory efficiency)
        max_grad_norm: Maximum gradient norm for clipping
        group_by_length: Group sequences by length for efficiency
        report_to: Reporting destination (tensorboard, wandb, none)
        remove_unused_columns: Whether to remove unused columns from dataset
    """
    from transformers import TrainingArguments
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        optim=optim,
        max_grad_norm=max_grad_norm,
        group_by_length=group_by_length,
        report_to=report_to,
        remove_unused_columns=remove_unused_columns,
    )

    print(f"Training config: {num_train_epochs} epochs, batch_size={per_device_train_batch_size}, lr={learning_rate}")
    print(f"Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")

    return training_args
