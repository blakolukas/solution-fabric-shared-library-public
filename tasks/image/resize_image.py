import cv2

from core.task import task


@task(
    outputs=["resized_image"],
    output_types={"resized_image": "image"},
    display_name="Resize Image",
    description="Resize an image to target dimensions",
    category="image",
    parameters={
        "image": {
            "type": "image",
            "required": True,
            "description": "Input image array",
        },
        "target_width": {
            "type": "int",
            "required": True,
            "description": "Target width in pixels",
        },
        "target_height": {
            "type": "int",
            "required": True,
            "description": "Target height in pixels",
        },
    },
    is_collapsed=True,
)
def resize_image(image, target_width: int, target_height: int):
    """
    Resize an image to target dimensions.

    Args:
        image: Input image array
        target_width: Target width in pixels
        target_height: Target height in pixels

    Returns:
        Resized image as numpy array
    """
    return cv2.resize(
        image, (target_width, target_height), interpolation=cv2.INTER_LINEAR
    )
