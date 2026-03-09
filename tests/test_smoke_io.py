"""Smoke tests for tasks/io/ — verify importability and basic execution."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# check_file_exists
# ---------------------------------------------------------------------------
class TestCheckFileExistsSmoke:
    """Smoke tests for tasks.io.check_file_exists."""

    def test_importable(self):
        from tasks.io.check_file_exists import check_file_exists

        assert hasattr(check_file_exists, "__wrapped_function__")

    @pytest.mark.unit
    def test_none_returns_none(self):
        from tasks.io.check_file_exists import check_file_exists

        result = check_file_exists.__wrapped_function__(None)
        assert result is None

    @pytest.mark.unit
    def test_existing_file(self, tmp_path):
        from tasks.io.check_file_exists import check_file_exists

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        result = check_file_exists.__wrapped_function__(str(test_file))
        import os

        assert os.path.isabs(result)
        assert result.endswith("test.txt")

    @pytest.mark.unit
    def test_missing_file_raises(self, tmp_path):
        from tasks.io.check_file_exists import check_file_exists

        missing = str(tmp_path / "nonexistent.txt")
        with pytest.raises(FileNotFoundError):
            check_file_exists.__wrapped_function__(missing)


# ---------------------------------------------------------------------------
# delete_file
# ---------------------------------------------------------------------------
class TestDeleteFileSmoke:
    """Smoke tests for tasks.io.delete_file."""

    def test_importable(self):
        from tasks.io.delete_file import delete_file

        assert hasattr(delete_file, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self, tmp_path):
        from tasks.io.delete_file import delete_file

        test_file = tmp_path / "to_delete.txt"
        test_file.write_text("bye")
        result = delete_file.__wrapped_function__(str(test_file))
        assert result is True
        assert not test_file.exists()


# ---------------------------------------------------------------------------
# generate_temp_filename
# ---------------------------------------------------------------------------
class TestGenerateTempFilenameSmoke:
    """Smoke tests for tasks.io.generate_temp_filename."""

    def test_importable(self):
        from tasks.io.generate_temp_filename import generate_temp_filename

        assert hasattr(generate_temp_filename, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.io.generate_temp_filename import generate_temp_filename

        result = generate_temp_filename.__wrapped_function__()
        assert isinstance(result, str)
        assert result.endswith(".png")
        assert result.startswith("temp_")

    @pytest.mark.unit
    def test_custom_extension(self):
        from tasks.io.generate_temp_filename import generate_temp_filename

        result = generate_temp_filename.__wrapped_function__(prefix="out", extension="jpg")
        assert result.endswith(".jpg")
        assert result.startswith("out_")

    @pytest.mark.unit
    def test_uniqueness(self):
        from tasks.io.generate_temp_filename import generate_temp_filename

        fn1 = generate_temp_filename.__wrapped_function__()
        fn2 = generate_temp_filename.__wrapped_function__()
        assert fn1 != fn2


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------
class TestLoadImageSmoke:
    """Smoke tests for tasks.io.load_image."""

    def test_importable(self):
        from tasks.io.load_image import load_image

        assert hasattr(load_image, "__wrapped_function__")

    @pytest.mark.unit
    def test_passthrough_array(self):
        from tasks.io.load_image import load_image

        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        result = load_image.__wrapped_function__(image=arr)
        np.testing.assert_array_equal(result, arr)

    @pytest.mark.unit
    def test_load_from_path_with_mock(self, tmp_path):
        """Mock cv2.imread to avoid needing a real image file."""
        fake_img = np.zeros((10, 10, 3), dtype=np.uint8)
        fake_path = str(tmp_path / "fake.png")
        with patch("cv2.imread", return_value=fake_img):
            from tasks.io.load_image import load_image

            result = load_image.__wrapped_function__(image_path=fake_path)
        np.testing.assert_array_equal(result, fake_img)

    @pytest.mark.unit
    def test_no_input_raises(self):
        from tasks.io.load_image import load_image

        with pytest.raises(ValueError):
            load_image.__wrapped_function__()


# ---------------------------------------------------------------------------
# load_video
# ---------------------------------------------------------------------------
class TestLoadVideoSmoke:
    """Smoke tests for tasks.io.load_video."""

    def test_importable(self):
        from tasks.io.load_video import load_video

        assert hasattr(load_video, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self, tmp_path):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        with patch("cv2.VideoCapture", return_value=mock_cap):
            from tasks.io.load_video import load_video

            result = load_video.__wrapped_function__("fake_video.mp4")
        assert result["video"] == "fake_video.mp4"
        assert result["video_capture"] is mock_cap

    @pytest.mark.unit
    def test_failed_open_raises(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        with patch("cv2.VideoCapture", return_value=mock_cap):
            from tasks.io.load_video import load_video

            with pytest.raises(RuntimeError):
                load_video.__wrapped_function__("bad_path.mp4")


# ---------------------------------------------------------------------------
# preview_video
# ---------------------------------------------------------------------------
class TestPreviewVideoSmoke:
    """Smoke tests for tasks.io.preview_video."""

    def test_importable(self):
        from tasks.io.preview_video import preview_video

        assert hasattr(preview_video, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.io.preview_video import preview_video

        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
        result = preview_video.__wrapped_function__(frames, fps=24.0, format="mp4")
        assert result["frame_count"] == 3
        assert result["fps"] == 24.0
        assert result["format"] == "mp4"

    @pytest.mark.unit
    def test_empty_frames_raises(self):
        from tasks.io.preview_video import preview_video

        with pytest.raises(ValueError):
            preview_video.__wrapped_function__([])


# ---------------------------------------------------------------------------
# save_image_to_disk
# ---------------------------------------------------------------------------
class TestSaveImageToDiskSmoke:
    """Smoke tests for tasks.io.save_image_to_disk."""

    def test_importable(self):
        from tasks.io.save_image_to_disk import save_image_to_disk

        assert hasattr(save_image_to_disk, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self, tmp_path):
        from tasks.io.save_image_to_disk import save_image_to_disk

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = save_image_to_disk.__wrapped_function__(
            img, save_path=str(tmp_path), save_name="out.png"
        )
        import os

        assert os.path.isfile(result)

    @pytest.mark.unit
    def test_non_array_raises(self):
        from tasks.io.save_image_to_disk import save_image_to_disk

        with pytest.raises(ValueError):
            save_image_to_disk.__wrapped_function__("not an array")


# ---------------------------------------------------------------------------
# save_images_to_directory
# ---------------------------------------------------------------------------
class TestSaveImagesToDirectorySmoke:
    """Smoke tests for tasks.io.save_images_to_directory."""

    def test_importable(self):
        from tasks.io.save_images_to_directory import save_images_to_directory

        assert hasattr(save_images_to_directory, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self, tmp_path):
        from PIL import Image

        from tasks.io.save_images_to_directory import save_images_to_directory

        pil_img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        result = save_images_to_directory.__wrapped_function__(
            [pil_img], output_dir=str(tmp_path), base_name="test"
        )
        import os

        assert len(result) == 1
        assert os.path.isfile(result[0])

    @pytest.mark.unit
    def test_multiple_images(self, tmp_path):
        from PIL import Image

        from tasks.io.save_images_to_directory import save_images_to_directory

        imgs = [Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)) for _ in range(3)]
        result = save_images_to_directory.__wrapped_function__(
            imgs, output_dir=str(tmp_path), base_name="img"
        )
        assert len(result) == 3


# ---------------------------------------------------------------------------
# save_text_to_file
# ---------------------------------------------------------------------------
class TestSaveTextToFileSmoke:
    """Smoke tests for tasks.io.save_text_to_file."""

    def test_importable(self):
        from tasks.io.save_text_to_file import save_text_to_file

        assert hasattr(save_text_to_file, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self, tmp_path):
        from tasks.io.save_text_to_file import save_text_to_file

        result = save_text_to_file.__wrapped_function__(
            "hello", directory=str(tmp_path), filename="output.txt"
        )
        import os

        assert os.path.isfile(result)
        with open(result, encoding="utf-8") as f:
            assert f.read() == "hello"
