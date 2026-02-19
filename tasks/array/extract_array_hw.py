import numpy as np

from core.task import task


@task(
    outputs=["orig_height", "orig_width"],
    display_name="Extract Array Height/Width",
    description="Extract height and width from a numpy array",
    category="array",
    parameters={
        "array": {
            "type": "ndarray",
            "required": True,
            "description": "Numpy array with at least 2 dimensions",
        },
    },
)
def extract_array_hw(array):
    """
    Extract original (H, W) from a numpy image-like array.

    Expects array shape (H, W, ...) or (H, W). Returns height/width as ints.

    Args:
        array: Numpy array with at least 2 dimensions

    Returns:
        Tuple of (height, width)
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("'array' must be a numpy array.")
    if array.ndim < 2:
        raise ValueError("'array' must have at least 2 dimensions (H, W, ...).")

    height, width = int(array.shape[0]), int(array.shape[1])

    return height, width
