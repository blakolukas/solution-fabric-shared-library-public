import numpy as np
import onnxruntime as ort

from core.logging import get_logger
from core.task import task

logger = get_logger(__name__)


@task(
    outputs=["output"],
    parameters={
        "onnx_session": {
            "type": "object",
            "required": True,
            "description": "ONNX runtime inference session",
        },
        "input_array": {
            "type": "ndarray",
            "required": True,
            "description": "Input array for the model",
        },
        "input_name": {
            "type": "str",
            "required": False,
            "default": "input",
            "description": "Name of the input node",
        },
    },
)
def onnx_inference(onnx_session: ort.InferenceSession, input_array: np.ndarray, input_name: str = "input"):
    """
    Run ONNX model inference on an input array with GPU optimization.

    Args:
        onnx_session: ONNX runtime inference session
        input_array: Input array for the model (should be contiguous for best performance)
        input_name: Name of the input node (default: "input")

    Returns:
        Model output as numpy array or list of arrays
    """
    # Run inference using IOBinding for better GPU performance
    try:
        io_binding = onnx_session.io_binding()

        # Bind input on CPU (GPU execution providers handle the transfer)
        io_binding.bind_cpu_input(input_name, input_array)

        # Get output names and bind outputs
        output_names = [output.name for output in onnx_session.get_outputs()]
        for output_name in output_names:
            io_binding.bind_output(output_name)

        # Run inference with IOBinding
        onnx_session.run_with_iobinding(io_binding)

        # Copy outputs back to CPU
        output = io_binding.copy_outputs_to_cpu()

    except Exception as e:
        logger.warning(f"IOBinding failed ({e}), falling back to standard inference")
        # Fallback to standard inference
        output = onnx_session.run(None, {input_name: input_array})

    # Return the raw model output - let workflows handle the format
    return output
