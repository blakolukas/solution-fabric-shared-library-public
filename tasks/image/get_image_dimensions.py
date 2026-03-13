from core.task import task


@task(
    outputs=["image_dimensions"],
    output_types={"image_dimensions": "json"},
    display_name="Get Image Dimensions",
    description="Get the height and width of an image",
    category="image",
    parameters={
        "image": {
            "type": "image",
            "required": True,
            "description": "Input image as numpy array",
        },
    },
    is_collapsed=True,
)
def get_image_dimensions(image):
    """
    Get the dimensions (height, width) of an image.

    Args:
        image: Input image as numpy array

    Returns:
        Tuple of (height, width)
    """
    height, width = image.shape[:2]
    return (height, width)
