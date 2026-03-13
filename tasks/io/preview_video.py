"""
Preview video task - displays/outputs video results from workflows.
"""

import numpy as np
from typing import List
from core.task import task


@task(
    outputs=["preview_video"],
    output_types={"preview_video": "video"},
    display_name="Preview Video",
    description="Output video frames for preview/display",
    category="io",
    parameters={
        "frames": {
            "type": "list",
            "required": True,
            "description": "List of video frames as numpy arrays",
        },
        "fps": {
            "type": "float",
            "default": 30.0,
            "description": "Frames per second for output video",
        },
        "format": {
            "type": "str",
            "default": "mp4",
            "description": "Output format: mp4, avi, or webm",
        },
    },
)
def preview_video(frames: List[np.ndarray], fps: float = 30.0, format: str = "mp4"):
    """
    Preview/output video frames from the workflow.

    This task serves as an output endpoint for video data in workflows.
    It takes a list of frames and packages them for output.

    Args:
        frames: List of video frames as numpy arrays (BGR format, HWC)
        fps: Frames per second for the output video
        format: Output format - 'mp4', 'avi', or 'webm'

    Returns:
        preview_video: Dictionary with video metadata and frames
    """
    if not frames:
        raise ValueError("No frames provided to preview")

    if not isinstance(frames, list):
        raise ValueError(f"Expected list of frames, got {type(frames)}")

    # Validate all frames are numpy arrays with consistent shape
    first_shape = None
    for i, frame in enumerate(frames):
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Frame {i} is not a numpy array: {type(frame)}")
        if first_shape is None:
            first_shape = frame.shape
        elif frame.shape != first_shape:
            raise ValueError(
                f"Frame {i} has inconsistent shape: {frame.shape} vs {first_shape}"
            )

    # Return video metadata and frames for the output system to encode
    return {
        "frames": frames,
        "fps": fps,
        "format": format,
        "frame_count": len(frames),
        "resolution": f"{first_shape[1]}x{first_shape[0]}" if first_shape else None,
    }
