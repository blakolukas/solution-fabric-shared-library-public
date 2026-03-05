from core.task import task


@task(
    outputs=["height", "width"],
    display_name="Extract H/W from Shape",
    description="Extract height and width from ONNX model input shape",
    category="array",
    parameters={
        "input_shape": {
            "type": "tuple",
            "required": True,
            "description": "Input shape tuple/list (expects N,C,H,W format)",
        },
    },
)
def extract_hw_from_shape(input_shape):
    """
    Extract height and width from ONNX model input shape.

    Args:
        input_shape: Input shape tuple/list (expects N,C,H,W format)

    Returns:
        Tuple of (height, width)
    """
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) < 4:
        raise ValueError("'input_shape' must be a list/tuple with at least 4 elements (N,C,H,W).")

    # Assume channel-first (N,C,H,W)
    height = input_shape[2]
    width = input_shape[3]

    if isinstance(height, str) or isinstance(width, str):
        # Some ONNX shapes may have symbolic dims; fail early for clarity
        raise ValueError(f"Cannot extract concrete H/W from symbolic shape: {input_shape}")

    return height, width
