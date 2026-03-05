"""
Camera Source - Frame generator for streaming camera processing.

Provides a frame generator that yields frames continuously from a camera device.
This is a streaming source task that produces a generator for use with
streaming workflows.
"""

import time
from typing import Any, Dict, Generator, Tuple

import cv2

from core.stream import StoppableGenerator
from core.task import task


@task(
    outputs=["frame_generator", "metadata"],
    output_types={"frame_generator": "generator", "metadata": "json"},
    display_name="Camera Source",
    description="Create a frame generator from a camera for streaming processing",
    category="camera",
    parameters={
        "camera_index": {
            "type": "int",
            "required": False,
            "default": 0,
            "description": "Camera device index",
        },
        "target_fps": {
            "type": "float",
            "required": False,
            "default": 30.0,
            "description": "Target frames per second",
        },
        "width": {
            "type": "int",
            "required": False,
            "default": 1280,
            "description": "Frame width",
        },
        "height": {
            "type": "int",
            "required": False,
            "default": 720,
            "description": "Frame height",
        },
    },
)
def camera_source(
    camera_index: int = 0,
    target_fps: float = 30.0,
    width: int = 1280,
    height: int = 720,
) -> Tuple[StoppableGenerator[Dict[str, Any]], dict]:
    """
    Create a frame generator from a camera device.

    This is a streaming source task - it produces a generator that yields
    frames continuously until stopped. Use this for streaming workflows
    where you want to process camera frames through a pipeline.

    Args:
        camera_index: Camera device index (default: 0)
        target_fps: Target frames per second (default: 30.0)
        width: Desired frame width (default: 1280)
        height: Desired frame height (default: 720)

    Returns:
        frame_generator: StoppableGenerator yielding numpy arrays (BGR, HWC)
        metadata: Camera metadata including fps, width, height
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {camera_index}")

    # Set desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_interval = 1.0 / target_fps if target_fps > 0 else 0

    metadata = {
        "source": "camera",
        "camera_index": camera_index,
        "target_fps": target_fps,
        "width": actual_width,
        "height": actual_height,
        "requested_width": width,
        "requested_height": height,
    }

    def _create_frame_iterator() -> Generator[Dict[str, Any], None, None]:
        """Internal generator function that yields camera frames."""
        last_frame_time = time.time()
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Brief pause and retry for webcam hiccups
                    time.sleep(0.01)
                    continue

                # Rate limiting to target FPS
                if frame_interval > 0:
                    elapsed = time.time() - last_frame_time
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
                    last_frame_time = time.time()

                frame_count += 1
                yield frame

        finally:
            cap.release()

    # Wrap in StoppableGenerator for cooperative cancellation support
    return StoppableGenerator(_create_frame_iterator()), metadata
