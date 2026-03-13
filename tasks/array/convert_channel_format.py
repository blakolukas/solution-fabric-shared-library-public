import numpy as np

from core.task import task


@task(
    outputs=["converted_array"],
    output_types={"converted_array": "array"},
    display_name="Convert Channel Format",
    description="Convert array between channel-first and channel-last formats",
    category="array",
    parameters={
        "array": {
            "type": "ndarray",
            "required": True,
            "description": "Input numpy array (must have at least 3 dimensions)",
        },
        "to_format": {
            "type": "str",
            "required": True,
            "description": "Target format ('channel_first' or 'channel_last')",
            "options": ["channel_first", "channel_last"],
        },
    },
)
def convert_channel_format(array, to_format: str):
    """
    Convert array between channel-first and channel-last formats.

    Args:
        array: Input numpy array (must have at least 3 dimensions)
        to_format: Target format ('channel_first' or 'channel_last')

    Returns:
        Converted array
    """
    if to_format not in ["channel_first", "channel_last"]:
        raise ValueError(
            "'to_format' must be either 'channel_first' or 'channel_last'."
        )

    if array.ndim < 3:
        raise ValueError("Input array must have at least 3 dimensions.")

    if to_format == "channel_first":
        # Convert from channel last (e.g., HWC) to channel first (e.g., CHW)
        return np.moveaxis(array, -1, 0)
    else:
        # Convert from channel first (e.g., CHW) to channel last (e.g., HWC)
        return np.moveaxis(array, 0, -1)
