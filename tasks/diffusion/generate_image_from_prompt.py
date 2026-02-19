from core.task import task


@task(
    outputs=["generated_images"],
    parameters={
        "pipeline": {
            "type": "object",
            "required": True,
            "description": "Stable Diffusion pipeline object",
        },
        "prompt": {
            "type": "str",
            "required": True,
            "multiline": True,
            "description": "Positive prompt describing what to generate",
        },
        "negative_prompt": {
            "type": "str",
            "required": False,
            "default": "",
            "multiline": True,
            "description": "Negative prompt describing what to avoid",
        },
        "num_inference_steps": {
            "type": "int",
            "required": False,
            "default": 50,
            "description": "Number of denoising steps",
        },
        "guidance_scale": {
            "type": "float",
            "required": False,
            "default": 7.5,
            "description": "Classifier-free guidance scale",
        },
        "width": {
            "type": "int",
            "required": False,
            "default": 512,
            "description": "Image width in pixels",
        },
        "height": {
            "type": "int",
            "required": False,
            "default": 512,
            "description": "Image height in pixels",
        },
        "seed": {
            "type": "int",
            "required": False,
            "default": None,
            "description": "Random seed for reproducibility",
        },
        "num_images_per_prompt": {
            "type": "int",
            "required": False,
            "default": 1,
            "description": "Total number of images to generate",
        },
        "batch_size": {
            "type": "int",
            "required": False,
            "default": 1,
            "description": "Number of images per batch",
        },
    },
)
def generate_image_from_prompt(
    pipeline,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int = None,
    num_images_per_prompt: int = 1,
    batch_size: int = 1,
):
    """
    Generate images using Stable Diffusion pipeline with batch support.

    Args:
        pipeline: Stable Diffusion pipeline object
        prompt: Positive prompt describing what to generate
        negative_prompt: Negative prompt describing what to avoid
        num_inference_steps: Number of denoising steps (default: 50)
        guidance_scale: Classifier-free guidance scale (default: 7.5)
        width: Image width in pixels (default: 512)
        height: Image height in pixels (default: 512)
        seed: Random seed for reproducibility (optional)
        num_images_per_prompt: Total number of images to generate (default: 1)
        batch_size: Number of images to generate per batch (default: 1)

    Returns:
        List of generated PIL Images
    """
    import torch
    
    all_images = []

    # Calculate number of batches needed
    num_batches = (num_images_per_prompt + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        # Calculate how many images in this batch
        remaining = num_images_per_prompt - len(all_images)
        current_batch_size = min(batch_size, remaining)

        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            device = pipeline.device
            # Use different seed for each batch to get varied results
            batch_seed = seed + batch_idx
            generator = torch.Generator(device=device).manual_seed(batch_seed)

        # Generate batch of images
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_images_per_prompt=current_batch_size,
            generator=generator,
        )

        # Add batch results to collection
        all_images.extend(result.images)

    return all_images
