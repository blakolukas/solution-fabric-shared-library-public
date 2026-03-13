import os

import cv2
import numpy as np

from core.task import task


@task(
    outputs=["saved_image_path"],
    output_types={"saved_image_path": "str"},
    display_name="Save Image to Disk",
    description="Save an image array to disk",
    category="io",
    parameters={
        "image_array": {
            "type": "image",
            "required": True,
            "description": "Numpy array containing the image",
        },
        "save_path": {
            "type": "str",
            "required": False,
            "default": ".",
            "description": "Directory path to save the image",
        },
        "save_name": {
            "type": "str",
            "required": False,
            "default": "output.png",
            "description": "Filename for the saved image",
        },
    },
    is_collapsed=True,
)
def save_image_to_disk(
    image_array, save_path: str = ".", save_name: str = "output.png"
):
    """
    Save an image array to disk.

    Args:
        image_array: Numpy array containing the image
        save_path: Directory path to save the image (default: current directory)
        save_name: Filename for the saved image (default: "output.png")

    Returns:
        Full path to the saved image

    Raises:
        ValueError: If image_array is not a numpy array
        IOError: If the image save operation fails
    """
    if not isinstance(image_array, np.ndarray):
        raise ValueError("'image_array' must be a numpy array.")

    abs_save_path = os.path.abspath(save_path)
    full_save_path = os.path.join(abs_save_path, save_name)

    success = cv2.imwrite(full_save_path, image_array)
    if not success:
        raise IOError(f"Failed to save image to: {full_save_path}")

    return full_save_path
