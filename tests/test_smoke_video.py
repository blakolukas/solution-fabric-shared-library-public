"""Smoke tests for tasks/video/ — verify importability and basic execution."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# video_source
# ---------------------------------------------------------------------------
class TestVideoSourceSmoke:
    """Smoke tests for tasks.video.video_source."""

    def test_importable(self):
        from tasks.video.video_source import video_source

        assert hasattr(video_source, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        from tasks.video.video_source import video_source

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # CAP_PROP_FPS=30, CAP_PROP_FRAME_COUNT=100, width=1280, height=720
        mock_cap.get.side_effect = [30.0, 100, 1280.0, 720.0]

        with patch("cv2.VideoCapture", return_value=mock_cap):
            frame_gen, metadata = video_source.__wrapped_function__(
                video_path="fake_video.mp4"
            )

        assert metadata["source"] == "video"
        assert metadata["path"] == "fake_video.mp4"
        assert metadata["native_fps"] == 30.0
        assert metadata["total_frames"] == 100
        assert metadata["width"] == 1280
        assert metadata["height"] == 720
        # frame_gen should be a StoppableGenerator (iterable)
        assert hasattr(frame_gen, "__iter__")

    @pytest.mark.unit
    def test_failed_open_raises(self):
        from tasks.video.video_source import video_source

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(RuntimeError, match="Failed to open video"):
                video_source.__wrapped_function__(video_path="nonexistent.mp4")

    @pytest.mark.unit
    def test_metadata_with_target_fps(self):
        from tasks.video.video_source import video_source

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [25.0, 250, 640.0, 480.0]

        with patch("cv2.VideoCapture", return_value=mock_cap):
            _, metadata = video_source.__wrapped_function__(
                video_path="clip.mp4", target_fps=10.0
            )

        assert metadata["target_fps"] == 10.0
        assert metadata["native_fps"] == 25.0
