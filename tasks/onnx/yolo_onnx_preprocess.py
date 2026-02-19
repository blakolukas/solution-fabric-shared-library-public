import cv2
import numpy as np
import onnxruntime as ort

from core.task import task


@task(
    outputs=["preprocessed_image", "original_shape", "input_name"],
    parameters={
        "image": {
            "type": "array",
            "required": True,
            "description": "Input image as numpy array (BGR format, HWC)",
        },
        "onnx_session": {
            "type": "object",
            "required": True,
            "description": "ONNX runtime session to get input requirements",
        },
        "target_size": {
            "type": "int",
            "required": False,
            "default": 640,
            "description": "Target size for YOLO model",
        },
    },
)
def yolo_onnx_preprocess(image: np.ndarray, onnx_session: ort.InferenceSession, target_size: int = 640):
    """
    Preprocess image for YOLO ONNX inference.
    Handles resizing, padding, normalization, and format conversion.

    Args:
        image: Input image as numpy array (BGR format, HWC)
        onnx_session: ONNX runtime session to get input requirements
        target_size: Target size for YOLO model (default: 640)

    Returns:
        Tuple of:
        - preprocessed_image: Preprocessed image ready for inference (NCHW format, float32)
        - original_shape: Original image shape (H, W) for postprocessing
        - input_name: Name of the model input node
    """
    # Get input name from the session
    input_name = onnx_session.get_inputs()[0].name

    # Store original shape for later use
    original_shape = image.shape[:2]  # (H, W)

    # Convert BGR to RGB (YOLO expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize with aspect ratio preservation (letterbox)
    h, w = image_rgb.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize image
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image (letterbox padding)
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)  # Gray padding

    # Calculate padding offsets to center the image
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2

    # Place resized image in center of padded canvas
    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    # Convert to float32 and normalize to [0, 1]
    preprocessed = padded.astype(np.float32) / 255.0

    # Convert from HWC to CHW format
    preprocessed = np.transpose(preprocessed, (2, 0, 1))

    # Add batch dimension (NCHW)
    preprocessed = np.expand_dims(preprocessed, axis=0)

    # Ensure contiguous array for optimal performance
    preprocessed = np.ascontiguousarray(preprocessed)

    return preprocessed, original_shape, input_name
