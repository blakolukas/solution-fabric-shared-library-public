import os

from core.task import task


@task(
    outputs=["saved_model_path"],
    parameters={
        "trained_model": {
            "type": "object",
            "required": True,
            "description": "The fine-tuned PEFT model",
        },
        "tokenizer": {
            "type": "object",
            "required": True,
            "description": "LLaMA tokenizer",
        },
        "output_dir": {
            "type": "str",
            "required": False,
            "default": "./llama_finetuned_final",
            "description": "Directory to save the model and tokenizer",
        },
    },
)
def save_llama_model(trained_model, tokenizer, output_dir: str = "./llama_finetuned_final"):
    """Save the fine-tuned LLaMA PEFT model and tokenizer to disk.

    Args:
        trained_model: The fine-tuned PEFT model
        tokenizer: LLaMA tokenizer to save alongside the model
        output_dir: Directory to save the model and tokenizer

    Returns:
        saved_model_path: Path where the model was saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the PEFT adapter weights
    trained_model.save_pretrained(output_dir)
    print(f"PEFT adapter saved to: {output_dir}")

    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to: {output_dir}")

    return output_dir
