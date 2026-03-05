"""
Video Source - Frame generator for streaming video file processing.

Provides a frame generator that yields frames from a video file.
This is a streaming source task that produces a generator for use with
streaming workflows.
"""

import time
from typing import Any, Dict, Generator, Optional, Tuple

import cv2

from core.task import task
from core.stream import StoppableGenerator


@task(
    outputs=["frame_generator", "metadata"],
    output_types={"frame_generator": "generator", "metadata": "json"},
    display_name="Video Source",
    description="Create a frame generator from a video file for streaming processing",
    category="video",
    parameters={
        "video_path": {
            "type": "str",
            "required": True,
            "description": "Path to video file",
        },
        "target_fps": {
            "type": "float",
            "required": False,
            "default": None,
            "description": "Target frames per second (None = use video FPS)",
        },
        "start_time": {
            "type": "float",
            "required": False,
            "default": 0.0,
            "description": "Start time in seconds",
        },
        "end_time": {
            "type": "float",
            "required": False,
            "default": None,
            "description": "End time in seconds (None = end of video)",
        },
        "loop": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Whether to loop the video",
        },
    },
)
def video_source(
    video_path: str,
    target_fps: Optional[float] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    loop: bool = False,
) -> Tuple[StoppableGenerator[Dict[str, Any]], dict]:
    """
    Create a frame generator from a video file.

    This is a streaming source task - it produces a generator that yields
    frames from a video file. Use this for streaming workflows where you
    want to process video frames through a pipeline.

    Args:
        video_path: Path to the video file
        target_fps: Target frames per second (None = use video's native FPS)
        start_time: Start time in seconds (default: 0.0)
        end_time: End time in seconds (None = end of video)
        loop: Whether to loop the video continuously (default: False)

    Returns:
        frame_generator: StoppableGenerator yielding numpy arrays (BGR, HWC)
        metadata: Video metadata including fps, duration, frame_count, width, height
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Get video metadata
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / native_fps if native_fps > 0 else 0

    # Use target FPS or native FPS
    fps = target_fps if target_fps is not None else native_fps
    frame_interval = 1.0 / fps if fps > 0 else 0

    metadata = {
        "source": "video",
        "path": video_path,
        "native_fps": native_fps,
        "target_fps": fps,
        "duration": duration,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "loop": loop,
    }

    def _create_frame_iterator() -> Generator[Dict[str, Any], None, None]:
        """Internal generator function that yields frames."""
        nonlocal cap

        # Seek to start time
        if start_time > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        last_frame_time = time.time()
        frame_count = 0
        effective_end_time = end_time if end_time is not None else duration

        try:
            while True:
                ret, frame = cap.read()
                current_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

                if not ret or (effective_end_time and current_pos >= effective_end_time):
                    if loop:
                        # Reset to start
                        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
                        continue
                    else:
                        break

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
