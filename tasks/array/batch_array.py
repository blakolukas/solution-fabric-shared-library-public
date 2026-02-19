import numpy as np

from core.task import task


@task(
    outputs=["batched_array"],
    display_name="Batch Array",
    description="Add a batch dimension to an array",
    category="array",
    parameters={
        "array": {
            "type": "ndarray",
            "required": True,
            "description": "Input numpy array",
        },
        "axis": {
            "type": "int",
            "required": False,
            "default": 0,
            "description": "Axis at which to add the batch dimension",
        },
    },
)
def batch_array(array, axis: int = 0) -> np.ndarray:
    """
    Add a batch dimension at the front of an array.

    Args:
        array: Input numpy array
        axis: Axis at which to add the batch dimension (default is 0)

    Returns:
        Array with batch dimension added at axis 0
    """
    return np.expand_dims(array, axis=axis)
