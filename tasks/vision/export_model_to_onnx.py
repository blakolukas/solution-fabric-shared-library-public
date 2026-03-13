from core.task import task


@task(
    outputs=["onnx_model_path"],
    output_types={"onnx_model_path": "str"},
    parameters={
        "yolo_model": {
            "type": "object",
            "required": True,
            "description": "Loaded YOLO model instance",
        },
        "imgsz": {
            "type": "int",
            "required": False,
            "default": 640,
            "description": "Input image size",
        },
        "simplify": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "Whether to simplify the ONNX model",
        },
    },
    is_collapsed=True,
)
def export_model_to_onnx(yolo_model, imgsz: int = 640, simplify: bool = True):
    """
    Export YOLO model to ONNX format.

    Args:
        yolo_model: Loaded YOLO model instance (from load_yolo_model task)
        imgsz: Input image size (default: 640)
        simplify: Whether to simplify the ONNX model (default: True)

    Returns:
        onnx_model_path: Path to the exported ONNX model
    """
    # Export to ONNX format with optimized settings
    # The export will create a .onnx file in the same directory as the .pt file
    export_path = yolo_model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=simplify,
        # Optimize for inference
        dynamic=False,  # Static shapes for better performance
        opset=17,  # Use ONNX opset 17 for broad compatibility
    )

    return str(export_path)
