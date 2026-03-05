"""
Load image task - provides image input to workflows.
"""

import cv2
import numpy as np

from core.task import task


@task(
    outputs=["image"],
    output_types={"image": "image"},
    display_name="Load Image",
    description="Load an image from file path or workflow input",
    category="io",
    parameters={
        "image_path": {
            "type": "str",
            "required": False,
            "description": "Path to image file. If not provided, expects 'image' from workflow inputs.",
        }
    },
)
def load_image(image_path: str = None, image: np.ndarray = None):
    """
    Load an image from file path or pass through from workflow input.

    This task serves as the entry point for image data in workflows.
    It can either:
    1. Load an image from a file path (for local/reference paths)
    2. Pass through an image provided directly as workflow input

    Args:
        image_path: Optional path to image file on disk
        image: Optional image array passed directly (takes precedence if both provided)

    Returns:
        image: Loaded/provided image as numpy array (BGR format, HWC)
    """
    # If image is directly provided (e.g., from multipart upload), use it
    if image is not None:
        if isinstance(image, np.ndarray):
            return image
        raise ValueError(f"Expected numpy array for image, got {type(image)}")

    # Otherwise, load from path
    if image_path:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from: {image_path}")

        return image

    raise ValueError("Either image_path or image must be provided")
