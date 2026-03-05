import numpy as np

from core.task import task


@task(
    outputs=["unbatched_array"],
    display_name="Unbatch Array",
    description="Remove the batch dimension from an array",
    category="array",
    parameters={
        "array": {
            "type": "ndarray",
            "required": True,
            "description": "Input numpy array with batch dimension",
        },
    },
)
def unbatch_array(array):
    """
    Remove the batch dimension from the front of an array.

    Args:
        array: Input numpy array with batch dimension

    Returns:
        Array without batch dimension
    """
    return np.squeeze(array, axis=0)
