from core.task import task


@task(
    outputs=["processed_dataset"],
    output_types={"processed_dataset": "object"},
    parameters={
        "dataset": {
            "type": "object",
            "required": True,
            "description": "Raw dataset object",
        },
        "tokenizer": {
            "type": "object",
            "required": True,
            "description": "LLaMA tokenizer for tokenization",
        },
        "max_length": {
            "type": "int",
            "required": False,
            "default": 2048,
            "description": "Maximum sequence length for tokenization",
        },
        "num_proc": {
            "type": "int",
            "required": False,
            "default": 4,
            "description": "Number of processes for parallel processing",
        },
    },
)
def prepare_llama_dataset_for_training(
    dataset, tokenizer, max_length: int = 2048, num_proc: int = 4
):
    """Prepare and tokenize ultrachat_200k dataset for LLaMA training.

    Args:
        dataset: Raw ultrachat_200k dataset object
        tokenizer: LLaMA tokenizer for tokenization
        max_length: Maximum sequence length for tokenization
        num_proc: Number of processes for parallel processing
    """

    def format_chat_template(example):
        """Format the ultrachat_200k messages into LLaMA chat format."""
        messages = example.get("messages", [])

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        return {"text": text}

    def tokenize_function(examples):
        """Tokenize the formatted text."""
        # Tokenize
        model_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None,  # Return lists, not tensors
        )

        # Set labels for causal language modeling
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    print("Formatting chat templates...")
    # Format the messages into chat format
    formatted_dataset = dataset.map(
        format_chat_template, num_proc=num_proc, desc="Formatting chat templates"
    )

    print("Tokenizing dataset...")
    # Apply tokenization
    processed_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=formatted_dataset.column_names,
        desc="Tokenizing",
    )

    print(f"Dataset prepared. Total examples: {len(processed_dataset)}")
    return processed_dataset
