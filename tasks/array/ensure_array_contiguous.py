import numpy as np

from core.task import task


@task(
    outputs=["contiguous_array"],
    display_name="Ensure Array Contiguous",
    description="Ensure that a numpy array is contiguous in memory",
    category="array",
    parameters={
        "array": {
            "type": "ndarray",
            "required": True,
            "description": "Input numpy array",
        },
    },
)
def ensure_array_contiguous(array):
    """
    Ensure that a numpy array is contiguous in memory for optimal performance.

    Arrays that are not C-contiguous can cause performance issues in ONNX inference
    and other numerical operations. This task converts arrays to contiguous format
    when needed, or passes them through unchanged if already contiguous.

    Args:
        array: Input numpy array

    Returns:
        Contiguous numpy array (same data, optimized memory layout)
    """
    if not array.flags.c_contiguous:
        return np.ascontiguousarray(array)
    return array
