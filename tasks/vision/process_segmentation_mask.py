import numpy as np

from core.task import task


@task(
    outputs=["processed_mask"],
    parameters={
        "mask_array": {
            "type": "ndarray",
            "required": True,
            "description": "Segmentation mask with class IDs",
        },
        "keep_class_id": {
            "type": "int",
            "required": False,
            "default": 15,
            "description": "Class ID to keep unblurred (default 15=person in Pascal VOC). Background=0.",
        },
    },
)
def process_segmentation_mask(mask_array, keep_class_id: int = 15):
    """
    Process a segmentation mask to create a binary mask for background blur.

    Converts a semantic segmentation mask where pixels have class IDs
    into a binary mask suitable for background blur (0=blur, 255=keep).

    Args:
        mask_array: Segmentation mask as numpy array (H, W) with class IDs
        keep_class_id: Class ID to keep unblurred

    Returns:
        Binary mask as numpy array (H, W) - 0 for blur regions, 255 for keep regions
    """
    if not isinstance(mask_array, np.ndarray):
        raise ValueError("'mask_array' must be a numpy array.")

    # Ensure mask is 2D (remove channel dimension if present)
    if len(mask_array.shape) == 3:
        mask_array = mask_array.squeeze()

    # Create binary mask: 255 where class matches (keep), 0 elsewhere (blur)
    binary_mask = np.where(mask_array == keep_class_id, 255, 0).astype(np.uint8)

    return binary_mask
