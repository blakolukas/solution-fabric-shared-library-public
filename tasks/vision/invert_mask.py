import cv2
import numpy as np

from core.task import task


@task(
    outputs=["inverted_mask"],
    display_name="Invert Mask",
    description="Invert a binary mask",
    category="vision",
    parameters={
        "mask_array": {
            "type": "ndarray",
            "required": True,
            "description": "Binary mask as numpy array",
        },
    },
)
def invert_mask(mask_array):
    """
    Invert a binary mask.

    Converts 0 to 255 and vice versa.

    Args:
        mask_array: Binary mask as numpy array

    Returns:
        Inverted binary mask as numpy array
    """
    if not isinstance(mask_array, np.ndarray):
        raise ValueError("'mask_array' must be a numpy array.")

    # Ensure mask is 2D (remove channel dimension if present)
    if len(mask_array.shape) == 3:
        mask_array = mask_array.squeeze()

    # Convert to uint8 if needed
    if mask_array.dtype != np.uint8:
        if mask_array.max() <= 1.0:
            mask_array = (mask_array * 255).astype(np.uint8)
        else:
            mask_array = mask_array.astype(np.uint8)

    # Invert the mask
    inverted_mask = cv2.bitwise_not(mask_array)

    return inverted_mask
