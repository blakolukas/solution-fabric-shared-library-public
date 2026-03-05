import numpy as np

from core.task import task

# Constants for postprocessing
EPSILON = 1e-6  # Small value to prevent division by zero in IoU calculation

# COCO dataset class names (80 classes)
COCO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@task(
    outputs=["detections"],
    parameters={
        "raw_output": {
            "type": "array",
            "required": True,
            "description": "Raw model output array or list",
        },
        "original_shape": {
            "type": "tuple",
            "required": True,
            "description": "Original image shape (H, W) before preprocessing",
        },
        "confidence_threshold": {
            "type": "float",
            "required": False,
            "default": 0.25,
            "description": "Minimum confidence threshold for detections",
        },
        "iou_threshold": {
            "type": "float",
            "required": False,
            "default": 0.45,
            "description": "IoU threshold for Non-Maximum Suppression",
        },
        "target_size": {
            "type": "int",
            "required": False,
            "default": 640,
            "description": "Model input size used during preprocessing",
        },
    },
)
def yolo_onnx_postprocess(
    raw_output: np.ndarray,
    original_shape: tuple,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    target_size: int = 640,
):
    """
    Postprocess YOLO ONNX model output to extract detections with NMS.

    Args:
        raw_output: Raw model output array or list (shape: [1, num_detections, 85+])
        original_shape: Original image shape (H, W) before preprocessing
        confidence_threshold: Minimum confidence threshold for detections (default: 0.25)
        iou_threshold: IoU threshold for Non-Maximum Suppression (default: 0.45)
        target_size: Model input size used during preprocessing (default: 640)

    Returns:
        detections: List of detection dictionaries containing:
            - bbox: [x1, y1, x2, y2] bounding box coordinates in original image space
            - class_name: Name of the detected class (using COCO class names)
            - confidence: Confidence score (0-1)
            - class_id: Integer class ID
    """
    detections = []

    # Handle list output from onnx_inference (extract first element)
    if isinstance(raw_output, list):
        if len(raw_output) == 0:
            return detections
        raw_output = raw_output[0]

    # Handle empty or invalid output
    if raw_output is None or len(raw_output.shape) < 2:
        return detections

    # Remove batch dimension if present
    if len(raw_output.shape) == 3 and raw_output.shape[0] == 1:
        raw_output = raw_output[0]

    # raw_output shape is typically (num_detections, 85) or (8400, 85)
    # Format: [x_center, y_center, width, height, confidence, class_0_prob, ..., class_79_prob]

    # For YOLOv8/v11 format: (num_detections, 4 + 80) where first 4 are bbox, rest are class scores
    # Transpose if needed to get (num_detections, features)
    if raw_output.shape[0] < raw_output.shape[1]:
        raw_output = raw_output.T

    # Extract bounding boxes and scores
    boxes = raw_output[:, :4]  # [x_center, y_center, width, height]
    scores = raw_output[:, 4:]  # Class scores for all classes

    # Get the maximum class score and class ID for each detection
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # Filter by confidence threshold
    mask = confidences >= confidence_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return detections

    # Convert from [x_center, y_center, w, h] to [x1, y1, x2, y2]
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    # Apply Non-Maximum Suppression (NMS)
    # Simple NMS implementation
    indices = nms(boxes_xyxy, confidences, iou_threshold)

    # Scale boxes back to original image size
    orig_h, orig_w = original_shape
    scale = min(target_size / orig_h, target_size / orig_w)
    pad_h = (target_size - int(orig_h * scale)) // 2
    pad_w = (target_size - int(orig_w * scale)) // 2

    for idx in indices:
        box = boxes_xyxy[idx]
        confidence = confidences[idx]
        class_id = class_ids[idx]

        # Remove padding and scale back to original size
        x1 = (box[0] - pad_w) / scale
        y1 = (box[1] - pad_h) / scale
        x2 = (box[2] - pad_w) / scale
        y2 = (box[3] - pad_h) / scale

        # Clip to image boundaries
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))

        # Get class name
        class_name = COCO_CLASS_NAMES[class_id] if class_id < len(COCO_CLASS_NAMES) else f"class_{class_id}"

        detections.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "class_name": class_name,
                "confidence": float(confidence),
                "class_id": int(class_id),
            }
        )

    return detections


def nms(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on detection boxes.

    Args:
        boxes: Array of shape (N, 4) with format [x1, y1, x2, y2]
        scores: Array of shape (N,) with confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        List of indices to keep
    """
    # Sort by score (highest first)
    sorted_indices = np.argsort(scores)[::-1]

    keep = []
    while len(sorted_indices) > 0:
        # Pick the box with highest score
        current = sorted_indices[0]
        keep.append(current)

        if len(sorted_indices) == 1:
            break

        # Get current box and remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[sorted_indices[1:]]

        # Calculate IoU between current box and remaining boxes
        ious = calculate_iou(current_box, remaining_boxes)

        # Keep only boxes with IoU less than threshold
        sorted_indices = sorted_indices[1:][ious < iou_threshold]

    return keep


def calculate_iou(box, boxes):
    """
    Calculate IoU between a box and multiple boxes.

    Args:
        box: Single box [x1, y1, x2, y2]
        boxes: Array of boxes (N, 4) with format [x1, y1, x2, y2]

    Returns:
        Array of IoU values (N,)
    """
    # Calculate intersection
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate union
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection

    # Calculate IoU
    iou = intersection / (union + EPSILON)

    return iou
