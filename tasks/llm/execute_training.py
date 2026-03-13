from core.task import task


@task(
    outputs=["training_result", "trained_model"],
    output_types={"training_result": "object", "trained_model": "object"},
    parameters={
        "trainer": {
            "type": "object",
            "required": True,
            "description": "Initialized HuggingFace Trainer instance",
        },
    },
)
def execute_training(trainer):
    """Execute the training loop and return results.

    Args:
        trainer: Initialized HuggingFace Trainer instance

    Returns:
        training_result: Training metrics and logs
        trained_model: The fine-tuned PEFT model
    """
    # Start training
    print("Starting training...")
    training_result = trainer.train()

    # Get the trained model
    trained_model = trainer.model

    print(f"Training completed. Final loss: {training_result.training_loss:.4f}")

    return training_result, trained_model
