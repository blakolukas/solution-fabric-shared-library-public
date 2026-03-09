"""Smoke tests for tasks/image/ — verify importability and basic execution."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# convert_bgr_to_rgb
# ---------------------------------------------------------------------------
class TestConvertBgrToRgbSmoke:
    """Smoke tests for tasks.image.convert_bgr_to_rgb."""

    def test_importable(self):
        from tasks.image.convert_bgr_to_rgb import convert_bgr_to_rgb

        assert hasattr(convert_bgr_to_rgb, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.image.convert_bgr_to_rgb import convert_bgr_to_rgb

        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # Blue channel
        result = convert_bgr_to_rgb.__wrapped_function__(bgr)
        assert result.shape == (4, 4, 3)
        # After BGR→RGB the old blue (index 0) is now red (index 2)
        assert result[0, 0, 2] == 255

    @pytest.mark.unit
    def test_grayscale_passthrough(self):
        from tasks.image.convert_bgr_to_rgb import convert_bgr_to_rgb

        gray = np.zeros((4, 4), dtype=np.uint8)
        result = convert_bgr_to_rgb.__wrapped_function__(gray)
        assert result.shape == (4, 4)


# ---------------------------------------------------------------------------
# convert_pil_to_numpy
# ---------------------------------------------------------------------------
class TestConvertPilToNumpySmoke:
    """Smoke tests for tasks.image.convert_pil_to_numpy."""

    def test_importable(self):
        from tasks.image.convert_pil_to_numpy import convert_pil_to_numpy

        assert hasattr(convert_pil_to_numpy, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from PIL import Image

        from tasks.image.convert_pil_to_numpy import convert_pil_to_numpy

        pil_img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8), mode="RGB")
        result = convert_pil_to_numpy.__wrapped_function__(pil_img)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10, 3)


# ---------------------------------------------------------------------------
# convert_to_grayscale
# ---------------------------------------------------------------------------
class TestConvertToGrayscaleSmoke:
    """Smoke tests for tasks.image.convert_to_grayscale."""

    def test_importable(self):
        from tasks.image.convert_to_grayscale import convert_to_grayscale

        assert hasattr(convert_to_grayscale, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.image.convert_to_grayscale import convert_to_grayscale

        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        result = convert_to_grayscale.__wrapped_function__(bgr)
        assert result.ndim == 2
        assert result.shape == (10, 10)


# ---------------------------------------------------------------------------
# ensure_directory_exists
# ---------------------------------------------------------------------------
class TestEnsureDirectoryExistsSmoke:
    """Smoke tests for tasks.image.ensure_directory_exists."""

    def test_importable(self):
        from tasks.image.ensure_directory_exists import ensure_directory_exists

        assert hasattr(ensure_directory_exists, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self, tmp_path):
        from tasks.image.ensure_directory_exists import ensure_directory_exists

        target = str(tmp_path / "new_dir")
        result = ensure_directory_exists.__wrapped_function__(target)
        import os

        assert os.path.isdir(result)

    @pytest.mark.unit
    def test_existing_directory(self, tmp_path):
        from tasks.image.ensure_directory_exists import ensure_directory_exists

        result = ensure_directory_exists.__wrapped_function__(str(tmp_path))
        import os

        assert os.path.isdir(result)


# ---------------------------------------------------------------------------
# get_image_dimensions
# ---------------------------------------------------------------------------
class TestGetImageDimensionsSmoke:
    """Smoke tests for tasks.image.get_image_dimensions."""

    def test_importable(self):
        from tasks.image.get_image_dimensions import get_image_dimensions

        assert hasattr(get_image_dimensions, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.image.get_image_dimensions import get_image_dimensions

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = get_image_dimensions.__wrapped_function__(img)
        assert result == (100, 200)


# ---------------------------------------------------------------------------
# min_max_normalize_image
# ---------------------------------------------------------------------------
class TestMinMaxNormalizeImageSmoke:
    """Smoke tests for tasks.image.min_max_normalize_image."""

    def test_importable(self):
        from tasks.image.min_max_normalize_image import min_max_normalize_image

        assert hasattr(min_max_normalize_image, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.image.min_max_normalize_image import min_max_normalize_image

        img = np.array([[0, 128, 255]], dtype=np.float32)
        result = min_max_normalize_image.__wrapped_function__(img)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    @pytest.mark.unit
    def test_custom_range(self):
        from tasks.image.min_max_normalize_image import min_max_normalize_image

        img = np.array([[0.0, 1.0, 2.0]])
        result = min_max_normalize_image.__wrapped_function__(img, min_value=0, max_value=255)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(255.0)


# ---------------------------------------------------------------------------
# resize_image
# ---------------------------------------------------------------------------
class TestResizeImageSmoke:
    """Smoke tests for tasks.image.resize_image."""

    def test_importable(self):
        from tasks.image.resize_image import resize_image

        assert hasattr(resize_image, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.image.resize_image import resize_image

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = resize_image.__wrapped_function__(img, target_width=50, target_height=50)
        assert result.shape == (50, 50, 3)

    @pytest.mark.unit
    def test_upscale(self):
        from tasks.image.resize_image import resize_image

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = resize_image.__wrapped_function__(img, target_width=200, target_height=150)
        assert result.shape == (150, 200, 3)
