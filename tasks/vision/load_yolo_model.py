from pathlib import Path

from core.task import task


@task(
    outputs=["yolo_model"],
    display_name="Load YOLO Model",
    description="Load YOLO v11 model for object detection",
    category="vision",
    parameters={
        "model_path": {
            "type": "str",
            "default": "yolo11n.pt",
            "description": "Path to the YOLO model file",
        },
    },
)
def load_yolo_model(model_path: str = "yolo11n.pt"):
    """
    Load YOLO v11 model using ultralytics library.

    Args:
        model_path: Path to the YOLO model file (e.g., "yolo11n.pt", "/path/to/custom_model.pt")

    Returns:
        yolo_model: Loaded YOLO model instance
    """
    import torch
    from ultralytics import YOLO
    
    model = YOLO(model_path)

    # Move model to GPU if available for faster inference
    if torch.cuda.is_available():
        model.to("cuda")

    return model
