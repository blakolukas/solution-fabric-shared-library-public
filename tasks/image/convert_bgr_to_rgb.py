import cv2
from core.task import task


@task(
    outputs=["rgb_image"],
    output_types={"rgb_image": "image"},
    display_name="BGR to RGB",
    description="Convert BGR image to RGB format",
    category="image",
    parameters={
        "image": {
            "type": "image",
            "required": True,
            "description": "Input image in BGR format (OpenCV default)",
        },
    },
)
def convert_bgr_to_rgb(image):
    """
    Convert BGR image to RGB format.

    Args:
        image: Input image in BGR format (OpenCV default)

    Returns:
        Image in RGB format
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
