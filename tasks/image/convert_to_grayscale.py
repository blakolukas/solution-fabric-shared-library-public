import cv2
from core.task import task


@task(
    outputs=["grayscale_image"],
    output_types={"grayscale_image": "image"},
    display_name="Convert to Grayscale",
    description="Convert a color image to grayscale",
    category="image",
    parameters={
        "image": {
            "type": "image",
            "required": True,
            "description": "Input color image as numpy array",
        },
    },
)
def convert_to_grayscale(image):
    """
    Convert a color image to grayscale for OCR preprocessing.

    Args:
        image: Input color image as numpy array

    Returns:
        Grayscale image as numpy array
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
