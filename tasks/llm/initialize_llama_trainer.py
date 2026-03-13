from core.task import task


@task(
    outputs=["trainer"],
    output_types={"trainer": "object"},
    parameters={
        "model": {
            "type": "object",
            "required": True,
            "description": "Base LLaMA model",
        },
        "lora_config": {
            "type": "object",
            "required": True,
            "description": "LoRA configuration",
        },
        "training_args": {
            "type": "object",
            "required": True,
            "description": "Training arguments",
        },
        "processed_dataset": {
            "type": "object",
            "required": True,
            "description": "Preprocessed training dataset",
        },
        "tokenizer": {
            "type": "object",
            "required": True,
            "description": "LLaMA tokenizer",
        },
    },
)
def initialize_llama_trainer(
    model, lora_config, training_args, processed_dataset, tokenizer
):
    """Initialize HuggingFace Trainer with PEFT model for LLaMA fine-tuning.

    Args:
        model: Base LLaMA model
        lora_config: LoRA configuration
        training_args: Training arguments
        processed_dataset: Preprocessed training dataset
        tokenizer: LLaMA tokenizer (used for data collation)
    """
    from peft import get_peft_model
    from transformers import DataCollatorForLanguageModeling, Trainer

    # Prepare model for training (handles gradient checkpointing properly)
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Apply LoRA to the model
    peft_model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(
        p.numel() for p in peft_model.parameters() if p.requires_grad
    )
    all_params = sum(p.numel() for p in peft_model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)"
    )

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )  # Causal LM, not masked LM

    # Create trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Trainer initialized successfully")
    return trainer
