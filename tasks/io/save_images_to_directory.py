import os

from core.task import task


@task(
    outputs=["saved_paths"],
    output_types={"saved_paths": "list"},
    parameters={
        "images": {
            "type": "list",
            "required": True,
            "description": "List of PIL Image objects to save",
        },
        "output_dir": {
            "type": "str",
            "required": True,
            "description": "Directory path to save images",
        },
        "base_name": {
            "type": "str",
            "required": False,
            "default": "generated",
            "description": "Base name for output files",
        },
        "format": {
            "type": "str",
            "required": False,
            "default": "png",
            "options": ["png", "jpg", "jpeg", "webp"],
            "description": "Image format",
        },
    },
    is_collapsed=True,
)
def save_images_to_directory(
    images: list, output_dir: str, base_name: str = "generated", format: str = "png"
):
    """
    Save a list of PIL Images to a directory with sequential naming.

    Args:
        images: List of PIL Image objects
        output_dir: Directory path to save images
        base_name: Base name for output files (default: "generated")
        format: Image format (png, jpg, etc.) (default: "png")

    Returns:
        List of saved file paths
    """
    saved_paths = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Handle single image case
    if len(images) == 1:
        filename = f"{base_name}.{format}"
        filepath = os.path.join(output_dir, filename)
        images[0].save(filepath)
        saved_paths.append(filepath)
    else:
        # Multiple images - use padded numbering
        num_digits = len(str(len(images)))

        for idx, image in enumerate(images):
            # Create padded filename (e.g., generated_001.png, generated_002.png)
            filename = f"{base_name}_{str(idx).zfill(num_digits)}.{format}"
            filepath = os.path.join(output_dir, filename)

            # Save image
            image.save(filepath)
            saved_paths.append(filepath)

    return saved_paths
