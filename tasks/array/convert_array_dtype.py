import numpy as np

from core.task import task


@task(
    outputs=["converted_array"],
    display_name="Convert Array Data Type",
    description="Convert array to a different data type",
    category="array",
    parameters={
        "array": {
            "type": "ndarray",
            "required": True,
            "description": "Input numpy array",
        },
        "target_dtype": {
            "type": "str",
            "required": True,
            "description": "Target data type (e.g., 'float32', 'int32', 'uint8')",
        },
    },
)
def convert_array_dtype(array, target_dtype: str):
    """
    Convert array to a different data type.

    Args:
        array: Input numpy array
        target_dtype: Target data type (e.g., 'float32', 'int32', 'uint8')

    Returns:
        Converted array
    """
    return array.astype(np.dtype(target_dtype))
