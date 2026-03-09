"""Smoke tests for tasks/camera/ — verify importability and basic execution."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# camera_source
# ---------------------------------------------------------------------------
class TestCameraSourceSmoke:
    """Smoke tests for tasks.camera.camera_source."""

    def test_importable(self):
        from tasks.camera.camera_source import camera_source

        assert hasattr(camera_source, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        from tasks.camera.camera_source import camera_source

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 1280.0  # returns same value for width and height

        with patch("cv2.VideoCapture", return_value=mock_cap):
            frame_gen, metadata = camera_source.__wrapped_function__(
                camera_index=0, target_fps=30.0, width=1280, height=720
            )

        assert metadata["source"] == "camera"
        assert metadata["camera_index"] == 0
        assert metadata["target_fps"] == 30.0
        # frame_gen should be a StoppableGenerator
        assert hasattr(frame_gen, "__iter__")

    @pytest.mark.unit
    def test_failed_open_raises(self):
        from tasks.camera.camera_source import camera_source

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(RuntimeError, match="Failed to open camera"):
                camera_source.__wrapped_function__(camera_index=99)


# ---------------------------------------------------------------------------
# camera_capture
# ---------------------------------------------------------------------------
class TestCameraCaptureSmoke:
    """Smoke tests for tasks.camera.camera_capture."""

    def test_importable(self):
        from tasks.camera.camera_capture import camera_capture

        assert hasattr(camera_capture, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        from tasks.camera.camera_capture import camera_capture

        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # read() returns (True, frame) - called for warmup + actual capture
        mock_cap.read.return_value = (True, fake_frame)
        mock_cap.get.return_value = 1280.0

        with patch("cv2.VideoCapture", return_value=mock_cap):
            frame, metadata = camera_capture.__wrapped_function__(
                camera_index=0, warmup_frames=1
            )

        assert isinstance(frame, np.ndarray)
        assert metadata["source"] == "camera"
        assert metadata["capture_mode"] == "single_frame"
        # Ensure camera is released after capture
        mock_cap.release.assert_called_once()

    @pytest.mark.unit
    def test_failed_open_raises(self):
        from tasks.camera.camera_capture import camera_capture

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(RuntimeError, match="Failed to open camera"):
                camera_capture.__wrapped_function__(camera_index=99)
