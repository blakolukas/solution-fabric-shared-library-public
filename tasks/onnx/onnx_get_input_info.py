from core.task import task


@task(
    outputs=["input_name", "input_shape", "input_type"],
    parameters={
        "onnx_session": {
            "type": "object",
            "required": True,
            "description": "ONNX runtime inference session",
        },
    },
)
def onnx_get_input_info(onnx_session):
    """
    Get input information from an ONNX model session.

    Args:
        onnx_session: ONNX runtime inference session

    Returns:
        Tuple of (input_name, input_shape, input_type)
    """
    input_name = onnx_session.get_inputs()[0].name
    input_shape = onnx_session.get_inputs()[0].shape
    input_type = onnx_session.get_inputs()[0].type

    return input_name, input_shape, input_type
