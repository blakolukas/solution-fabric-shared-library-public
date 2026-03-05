"""
Camera Capture - Single-frame capture from camera.

Captures a single frame from a camera device and immediately releases
the camera resource. This is suitable for one-shot capture scenarios where
you need a single photo/snapshot without continuous streaming.

For continuous frame processing, use camera_source which provides a streaming
generator optimized for high-throughput video processing.
"""

from typing import Tuple

import cv2
import numpy as np

from core.task import task


@task(
    outputs=["frame", "metadata"],
    output_types={"frame": "image", "metadata": "json"},
    display_name="Camera Capture",
    description="Capture a single frame from camera (one-shot)",
    category="camera",
    parameters={
        "camera_index": {
            "type": "int",
            "required": False,
            "default": 0,
            "description": "Camera device index",
        },
        "width": {
            "type": "int",
            "required": False,
            "default": 1280,
            "description": "Desired frame width",
        },
        "height": {
            "type": "int",
            "required": False,
            "default": 720,
            "description": "Desired frame height",
        },
        "warmup_frames": {
            "type": "int",
            "required": False,
            "default": 3,
            "description": "Number of frames to skip for camera warmup",
        },
    },
)
def camera_capture(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    warmup_frames: int = 3,
) -> Tuple[np.ndarray, dict]:
    """
    Capture a single frame from a camera device.

    This task opens the camera, captures one frame, and immediately releases
    the camera resource. The camera initialization adds ~200-500ms latency,
    making this suitable for one-shot captures but not continuous streaming.

    Note: Camera initialization latency (~200-500ms) makes this task suitable
    for periodic/one-shot captures. For continuous frame processing, use
    camera_source which keeps the camera open for streaming.

    Args:
        camera_index: Camera device index (default: 0)
        width: Desired frame width (default: 1280)
        height: Desired frame height (default: 720)
        warmup_frames: Number of frames to skip for auto-exposure/white-balance (default: 3)

    Returns:
        frame: Captured frame as numpy array (BGR, HWC format)
        metadata: Camera metadata including actual width, height, and source info
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {camera_index}")

    try:
        # Set desired resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Skip warmup frames for auto-exposure and white balance stabilization
        for _ in range(warmup_frames):
            ret, _ = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read warmup frames from camera {camera_index}")

        # Capture the actual frame
        ret, frame = cap.read()

        if not ret:
            raise RuntimeError(f"Failed to capture frame from camera {camera_index}")

        # Get actual resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        metadata = {
            "source": "camera",
            "camera_index": camera_index,
            "width": actual_width,
            "height": actual_height,
            "requested_width": width,
            "requested_height": height,
            "capture_mode": "single_frame",
        }

        return frame, metadata

    finally:
        # Always release camera resource
        cap.release()
