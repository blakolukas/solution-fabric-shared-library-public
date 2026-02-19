import json

import tasks  # Import to trigger task registration
from core.logging import init_logging
from core.workflow import Workflow


def main():
    init_logging(level="DEBUG", log_file="fabricflow.log")

    with open("workflows/stable_diffusion.json", "r") as f:
        workflow_json = json.load(f)

    wf = Workflow.from_definition(workflow_json)

    # Example: Generate 4 images in batches of 2 (like AUTOMATIC1111)
    initial_context = {
        "inputs": {
            # Model configuration
            "model_id": "runwayml/stable-diffusion-v1-5",
            "device": "cuda",  # Use "cpu" if CUDA is not available
            # Generation parameters
            "prompt": "a beautiful landscape with mountains and a lake, sunset, highly detailed, 8k",
            "negative_prompt": "blurry, low quality, distorted, ugly, bad anatomy",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512,
            "seed": 42,  # For reproducibility
            # Batch generation parameters
            "num_images_per_prompt": 8,  # Total number of images to generate
            "batch_size": 2,  # Generate 2 images at a time (for memory management)
            # Output configuration
            "output_dir": "output/",
            "output_name": "landscape",
            "output_format": "png",
        }
    }

    print("Starting Stable Diffusion workflow...")
    print(f"Prompt: {initial_context['inputs']['prompt']}")
    print(f"Model: {initial_context['inputs']['model_id']}")
    print(f"Steps: {initial_context['inputs']['num_inference_steps']}")
    print(
        f"Generating {initial_context['inputs']['num_images_per_prompt']} images in batches of {initial_context['inputs']['batch_size']}..."
    )

    result = wf.run(initial_context)

    print("\nWorkflow completed successfully!")
    print(f"Generated images saved to: {initial_context['inputs']['output_dir']}")
    for path in result.get("saved_image_paths", []):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
