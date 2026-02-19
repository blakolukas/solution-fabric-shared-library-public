from core.task import task


@task(
    outputs=["pipeline"],
    parameters={
        "model_id": {
            "type": "str",
            "required": False,
            "default": "runwayml/stable-diffusion-v1-5",
            "description": "HuggingFace model ID",
        },
        "device": {
            "type": "str",
            "required": False,
            "default": "cuda",
            "options": ["cuda", "cpu"],
            "description": "Device to load model on",
        },
    },
)
def load_stable_diffusion_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda"):
    """
    Load a Stable Diffusion pipeline from HuggingFace.

    Args:
        model_id: HuggingFace model ID (default: runwayml/stable-diffusion-v1-5)
        device: Device to load model on (cuda/cpu)

    Returns:
        Stable Diffusion pipeline object
    """
    import torch
    from diffusers import StableDiffusionPipeline
    
    # Check if CUDA is available, fallback to CPU if not
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,  # Disable safety checker for speed
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(device)

    # Enable memory optimizations
    if device == "cuda":
        try:
            pipeline.enable_attention_slicing()
        except Exception:
            pass

    return pipeline
