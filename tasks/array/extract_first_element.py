"""Extract first element from a list or tuple."""

import numpy as np

from core.task import task


@task(
    outputs=["first_element"],
    display_name="Get First Element",
    description="Extract the first element from a list or tuple",
    category="array",
    parameters={
        "output_list": {
            "type": "any",
            "required": True,
            "description": "List or tuple to extract first element from",
        },
    },
)
def extract_first_element(output_list):
    """
    Extract the first element from a list or tuple.

    This is a generic array manipulation task useful for extracting
    single elements from list outputs (ONNX inference, image generation, etc.).

    Args:
        output_list: List/tuple to extract first element from

    Returns:
        First element from the list/tuple
    """
    if isinstance(output_list, (list, tuple)) and len(output_list) > 0:
        return output_list[0]
    elif isinstance(output_list, np.ndarray):
        # Already a single array
        return output_list
    else:
        raise ValueError(f"Expected list/tuple/ndarray, got {type(output_list)}")
