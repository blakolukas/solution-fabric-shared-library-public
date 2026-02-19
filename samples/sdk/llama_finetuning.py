"""
LLaMA Fine-tuning Client Script

This script fine-tunes meta-llama/Meta-Llama-3-8B-Instruct using the HuggingFaceH4/ultrachat_200k dataset.
All configuration is defined in the workflow JSON file.
"""

import json

from core.logging import init_logging
from core.workflow import Workflow


def main():
    init_logging(level="DEBUG", log_file="fabricflow.log")

    # Load workflow definition
    with open("workflows/llama_finetuning.json") as f:
        workflow_def = json.load(f)

    # Create workflow
    workflow = Workflow.from_definition(workflow_def)

    # Run workflow (all parameters are in the workflow JSON)
    print("Starting LLaMA 3 fine-tuning workflow...")
    print("=" * 80)

    result = workflow.run({})

    # Print results
    print("=" * 80)
    print("Fine-tuning completed!")
    print(f"Model saved to: {result.get('saved_model_path')}")

    training_result = result.get("training_result")
    if training_result:
        print(f"Final training loss: {training_result.training_loss:.4f}")

    return result


if __name__ == "__main__":
    main()
