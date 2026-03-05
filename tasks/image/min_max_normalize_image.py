import numpy as np

from core.task import task


@task(
    outputs=["normalized_image_array"],
    display_name="Min-Max Normalize Image",
    description="Normalize image array to [0, 1] range using min-max normalization",
    category="image",
    parameters={
        "image_array": {
            "type": "ndarray",
            "required": True,
            "description": "Input image as numpy array",
        },
        "min_value": {
            "type": "float",
            "required": False,
            "default": 0,
            "description": "Minimum value for normalization range",
        },
        "max_value": {
            "type": "float",
            "required": False,
            "default": 1,
            "description": "Maximum value for normalization range",
        },
    },
)
def min_max_normalize_image(image_array, min_value: float = 0, max_value: float = 1):
    """
    Normalize image array to specified range using min-max normalization.

    Args:
        image_array: Input image as numpy array
        min_value: Minimum value for normalization range (default: 0)
        max_value: Maximum value for normalization range (default: 1)

    Returns:
        Normalized image as numpy array (values scaled to [min_value, max_value])
    """
    arr_min = np.min(image_array)
    arr_max = np.max(image_array)
    normalized = (image_array - arr_min) / (arr_max - arr_min)
    return normalized * (max_value - min_value) + min_value
