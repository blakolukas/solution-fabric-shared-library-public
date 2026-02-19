import numpy as np

from core.task import task


@task(
    outputs=["image_array"],
    output_types={"image_array": "image"},
    display_name="PIL to NumPy",
    description="Convert PIL Image to numpy array for processing",
    category="image",
    parameters={
        "pil_image": {
            "type": "image",
            "required": True,
            "description": "PIL Image object to convert",
        },
    },
)
def convert_pil_to_numpy(pil_image):
    """
    Convert PIL Image to numpy array.

    This task converts a PIL Image object to a numpy array (BGR format)
    suitable for OpenCV and other image processing tasks.

    Args:
        pil_image: PIL Image object

    Returns:
        Numpy array representation of the image (BGR format, HWC)
    """
    import cv2

    # Convert PIL to numpy (RGB)
    rgb_array = np.array(pil_image)

    # Convert RGB to BGR for OpenCV compatibility
    if len(rgb_array.shape) == 3 and rgb_array.shape[2] >= 3:
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array

    return rgb_array
