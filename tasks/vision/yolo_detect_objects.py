import numpy as np

from core.task import task


@task(
    outputs=["detections"],
    output_types={"detections": "json"},
    is_collapsed=True,
    display_name="Detect Objects (YOLO)",
    description="Perform object detection using YOLO model",
    category="vision",
    parameters={
        "yolo_model": {
            "type": "object",
            "required": True,
            "description": "Loaded YOLO model instance",
        },
        "image": {
            "type": "image",
            "required": True,
            "description": "Input image as numpy array (BGR format, HWC)",
        },
        "confidence_threshold": {
            "type": "float",
            "required": False,
            "default": 0.25,
            "min": 0.0,
            "max": 1.0,
            "description": "Minimum confidence threshold for detections",
        },
    },
)
def yolo_detect_objects(yolo_model, image: np.ndarray, confidence_threshold: float = 0.25):
    """
    Perform object detection using YOLO model.

    Args:
        yolo_model: Loaded YOLO model instance
        image: Input image as numpy array (BGR format, HWC)
        confidence_threshold: Minimum confidence threshold for detections (default: 0.25)

    Returns:
        detections: List of detection dictionaries containing:
            - bbox: [x1, y1, x2, y2] bounding box coordinates
            - class_name: Name of the detected class
            - confidence: Confidence score (0-1)
            - class_id: Integer class ID
    """
    # Run inference with optimizations:
    # - stream=True for generator (reduces memory)
    # - verbose=False to suppress output
    # - conf sets confidence threshold
    results = yolo_model(image, verbose=False, conf=confidence_threshold, stream=True)

    detections = []

    # Process results
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            continue

        # Batch process all boxes at once for efficiency
        xyxy = boxes.xyxy.cpu().numpy()  # [N, 4]
        confidences = boxes.conf.cpu().numpy()  # [N]
        class_ids = boxes.cls.cpu().numpy().astype(int)  # [N]

        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i]
            class_id = class_ids[i]
            confidence = confidences[i]
            class_name = result.names[class_id]

            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                }
            )

    return detections
