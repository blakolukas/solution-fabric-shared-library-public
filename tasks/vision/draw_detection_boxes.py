import cv2
import numpy as np

from core.task import task


@task(
    outputs=["annotated_image"],
    output_types={"annotated_image": "image"},
    display_name="Draw Detection Boxes",
    description="Draw bounding boxes with class names and confidence scores on image",
    category="vision",
    parameters={
        "image": {
            "type": "image",
            "required": True,
            "description": "Input image as numpy array (BGR format, HWC)",
        },
        "detections": {
            "type": "list",
            "required": True,
            "description": "List of detection dictionaries from YOLO",
        },
    },
)
def draw_detection_boxes(image: np.ndarray, detections: list):
    """
    Draw bounding boxes with class names and confidence scores on image.

    Args:
        image: Input image as numpy array (BGR format, HWC)
        detections: List of detection dictionaries containing:
            - bbox: [x1, y1, x2, y2] bounding box coordinates
            - class_name: Name of the detected class
            - confidence: Confidence score (0-1)

    Returns:
        annotated_image: Image with drawn bounding boxes and labels
    """
    # Create a copy to avoid modifying the original
    annotated_image = image.copy()

    # Define colors for different classes (cycling through a color palette)
    colors = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]

    for detection in detections:
        bbox = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        class_id = detection["class_id"]

        # Get coordinates
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        # Select color based on class_id
        color = colors[class_id % len(colors)]

        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        label = f"{class_name}: {confidence:.2f}"

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Draw background rectangle for text
        cv2.rectangle(
            annotated_image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            annotated_image,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White text
            2,
            cv2.LINE_AA,
        )

    return annotated_image
