"""
Load video task - provides video input to workflows.
"""

import cv2

from core.task import task


@task(
    outputs=["video_capture", "video"],
    output_types={"video_capture": "object", "video": "video"},
    display_name="Load Video",
    description="Load a video file for processing",
    category="io",
    parameters={
        "video_path": {
            "type": "str",
            "required": True,
            "description": "Path to video file",
        },
    },
)
def load_video(video_path: str):
    """
    Load a video file for frame-by-frame processing.

    This task creates a cv2.VideoCapture object for the specified video file.
    The capture object can be used by downstream tasks to process frames.

    Args:
        video_path: Path to video file

    Returns:
        video_capture: OpenCV VideoCapture object
        video: Path to the video file (for preview)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    return {"video_capture": cap, "video": video_path}
