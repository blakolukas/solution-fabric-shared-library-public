import os

import onnxruntime as ort

from core.task import task


@task(
    outputs=["onnx_session"],
    output_types={"onnx_session": "object"},
    parameters={
        "onnx_model_path": {
            "type": "str",
            "required": True,
            "description": "Path to the ONNX model file",
        },
        "device_id": {
            "type": "int",
            "required": False,
            "default": 0,
            "description": "GPU device ID to use (for multi-GPU systems)",
        },
        "execution_mode": {
            "type": "str",
            "required": False,
            "default": "parallel",
            "description": "Execution mode - 'parallel' or 'sequential'",
        },
    },
)
def onnx_load_model(
    onnx_model_path: str, device_id: int = 0, execution_mode: str = "parallel"
) -> ort.InferenceSession:
    """
    Load an ONNX model and create an inference session with optimizations.
    Supports DirectML (Windows), CUDA (Linux/Windows), and CPU execution.

    Args:
        onnx_model_path: Path to the ONNX model file
        device_id: GPU device ID to use (default: 0, for multi-GPU systems)
        execution_mode: Execution mode - "parallel" or "sequential" (default: "parallel")
                       Use "sequential" for DirectML on Windows

    Returns:
        ONNX runtime inference session
    """
    abs_path = os.path.abspath(onnx_model_path)

    # Configure session options for maximum performance
    so = ort.SessionOptions()

    # Suppress ONNX Runtime warnings (like memcpy warnings)
    so.log_severity_level = 3  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal

    # Enable all graph optimizations
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Enable memory optimizations
    so.enable_cpu_mem_arena = True
    so.enable_mem_pattern = True
    so.enable_mem_reuse = True

    # Set execution mode based on parameter or environment variable
    # Read from environment variable if available, otherwise use the parameter
    env_execution_mode = os.getenv("ONNX_EXECUTION_MODE", execution_mode)
    if env_execution_mode.lower() == "sequential":
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    else:
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    # Try GPU providers first, then fall back to optimized CPU
    try:
        # Check available providers
        available_providers = ort.get_available_providers()
        providers = []

        # Configure DirectML provider (Windows, all DirectX 12 GPUs)
        if "DmlExecutionProvider" in available_providers:
            directml_options = {
                "device_id": device_id,
                "enable_dynamic_graph_fusion": True,
                "disable_metacommands": False,
            }
            providers.append(("DmlExecutionProvider", directml_options))

        # Configure CUDA provider (NVIDIA GPUs)
        elif "CUDAExecutionProvider" in available_providers:
            cuda_options = {
                "device_id": device_id,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 6 * 1024 * 1024 * 1024,  # 6GB limit
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
                "cudnn_conv_use_max_workspace": "1",
                "enable_cuda_graph": "0",
            }
            providers.append(("CUDAExecutionProvider", cuda_options))

        # Always add CPU as fallback
        providers.append("CPUExecutionProvider")

        # Create session with configured providers
        session = ort.InferenceSession(abs_path, sess_options=so, providers=providers)

        return session

    except Exception:
        # Fall back to CPU with optimizations
        # Read thread configuration from environment variables
        intra_threads = int(os.getenv("ONNX_INTRA_OP_NUM_THREADS", "0"))
        inter_threads = int(os.getenv("ONNX_INTER_OP_NUM_THREADS", "0"))

        so.intra_op_num_threads = intra_threads  # 0 = use all available cores
        so.inter_op_num_threads = inter_threads  # 0 = use all available cores

        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(abs_path, sess_options=so, providers=providers)

        return session
