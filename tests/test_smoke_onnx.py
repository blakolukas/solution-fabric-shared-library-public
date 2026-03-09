"""Smoke tests for tasks/onnx/ — verify importability and basic execution."""

import sys
import zipfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_ort_mock():
    """Create a fresh onnxruntime mock with required attributes."""
    mock_ort = MagicMock()
    # Use MagicMock() instances (not the class) to avoid spec issues
    # when InferenceSession is called with positional args in task code.
    mock_ort.InferenceSession = MagicMock()
    mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
    mock_ort.ExecutionMode.ORT_SEQUENTIAL = 0
    mock_ort.ExecutionMode.ORT_PARALLEL = 1
    mock_ort.SessionOptions = MagicMock()
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
    return mock_ort


@pytest.fixture(autouse=True)
def _patch_onnxruntime():
    """Ensure onnxruntime is mocked for all tests in this module.

    Creates a fresh mock for each test and manually patches only the
    'onnxruntime' key in sys.modules to avoid disturbing other cached modules.
    """
    mock_ort = _make_ort_mock()
    had_key = "onnxruntime" in sys.modules
    original = sys.modules.get("onnxruntime")
    sys.modules["onnxruntime"] = mock_ort
    yield mock_ort
    if had_key:
        sys.modules["onnxruntime"] = original
    else:
        sys.modules.pop("onnxruntime", None)


# ---------------------------------------------------------------------------
# onnx_load_model
# ---------------------------------------------------------------------------
class TestOnnxLoadModelSmoke:
    """Smoke tests for tasks.onnx.onnx_load_model."""

    def test_importable(self):
        from tasks.onnx.onnx_load_model import onnx_load_model
        assert hasattr(onnx_load_model, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self, tmp_path):
        from tasks.onnx.onnx_load_model import onnx_load_model

        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake-onnx-content")

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = []
        mock_session.get_outputs.return_value = []

        # Patch where the symbol is USED, not where it is defined
        with patch("tasks.onnx.onnx_load_model.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_session
            mock_ort.SessionOptions.return_value = MagicMock()
            mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99

            result = onnx_load_model.__wrapped_function__(
                onnx_model_path=str(model_file)
            )

        assert result is mock_session

        
# ---------------------------------------------------------------------------
# onnx_model_get_input_info  (onnx_get_input_info)
# ---------------------------------------------------------------------------
class TestOnnxModelGetInputInfoSmoke:
    """Smoke tests for tasks.onnx.onnx_model_get_input_info."""

    def test_importable(self):
        from tasks.onnx.onnx_model_get_input_info import onnx_get_input_info

        assert hasattr(onnx_get_input_info, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.onnx.onnx_model_get_input_info import onnx_get_input_info

        mock_input = MagicMock()
        mock_input.name = "images"
        mock_input.shape = [1, 3, 640, 640]
        mock_input.type = "tensor(float)"

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]

        name, shape, dtype = onnx_get_input_info.__wrapped_function__(
            onnx_session=mock_session
        )
        assert name == "images"
        assert shape == [1, 3, 640, 640]
        assert dtype == "tensor(float)"


# ---------------------------------------------------------------------------
# onnx_image_inference  (onnx_inference)
# ---------------------------------------------------------------------------
class TestOnnxImageInferenceSmoke:
    """Smoke tests for tasks.onnx.onnx_image_inference."""

    def test_importable(self):
        from tasks.onnx.onnx_image_inference import onnx_inference

        assert hasattr(onnx_inference, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        from tasks.onnx.onnx_image_inference import onnx_inference

        expected_output = [np.zeros((1, 84, 8400), dtype=np.float32)]

        # Use standard run() path (IOBinding may fail in mocked env)
        mock_session = MagicMock()
        mock_session.io_binding.side_effect = Exception("no IOBinding in test")
        mock_session.run.return_value = expected_output

        mock_output_node = MagicMock()
        mock_output_node.name = "output0"
        mock_session.get_outputs.return_value = [mock_output_node]

        input_array = np.zeros((1, 3, 640, 640), dtype=np.float32)
        result = onnx_inference.__wrapped_function__(
            onnx_session=mock_session,
            input_array=input_array,
            input_name="images",
        )
        assert result == expected_output


# ---------------------------------------------------------------------------
# extract_onnx_model
# ---------------------------------------------------------------------------
class TestExtractOnnxModelSmoke:
    """Smoke tests for tasks.onnx.extract_onnx_model."""

    def test_importable(self):
        from tasks.onnx.extract_onnx_model import extract_onnx_model

        assert hasattr(extract_onnx_model, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_extraction(self, tmp_path):
        from tasks.onnx.extract_onnx_model import extract_onnx_model

        # Create a fake zip with model.onnx inside
        zip_path = tmp_path / "model.zip"
        with zipfile.ZipFile(str(zip_path), "w") as zf:
            zf.writestr("model.onnx", b"fake onnx content")

        result = extract_onnx_model.__wrapped_function__(
            model_path=str(zip_path),
            repository_id="org/my-model",
            base_directory=str(tmp_path / "models"),
        )
        import os

        assert result.endswith("model.onnx")
        assert os.path.isfile(result)

    @pytest.mark.unit
    def test_existing_model_skips_extraction(self, tmp_path):
        from tasks.onnx.extract_onnx_model import extract_onnx_model

        # Pre-create the model directory and model.onnx
        model_dir = tmp_path / "models" / "my-model"
        model_dir.mkdir(parents=True)
        existing = model_dir / "model.onnx"
        existing.write_bytes(b"cached model")

        result = extract_onnx_model.__wrapped_function__(
            model_path="irrelevant.zip",
            repository_id="org/my-model",
            base_directory=str(tmp_path / "models"),
        )
        assert str(result) == str(existing)


# ---------------------------------------------------------------------------
# yolo_onnx_preprocess
# ---------------------------------------------------------------------------
class TestYoloOnnxPreprocessSmoke:
    """Smoke tests for tasks.onnx.yolo_onnx_preprocess."""

    def test_importable(self):
        from tasks.onnx.yolo_onnx_preprocess import yolo_onnx_preprocess

        assert hasattr(yolo_onnx_preprocess, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.onnx.yolo_onnx_preprocess import yolo_onnx_preprocess

        mock_input = MagicMock()
        mock_input.name = "images"
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        preprocessed, original_shape, input_name = yolo_onnx_preprocess.__wrapped_function__(
            image=image, onnx_session=mock_session, target_size=640
        )

        assert preprocessed.shape == (1, 3, 640, 640)
        assert preprocessed.dtype == np.float32
        assert original_shape == (480, 640)
        assert input_name == "images"


# ---------------------------------------------------------------------------
# yolo_onnx_postprocess
# ---------------------------------------------------------------------------
class TestYoloOnnxPostprocessSmoke:
    """Smoke tests for tasks.onnx.yolo_onnx_postprocess."""

    def test_importable(self):
        from tasks.onnx.yolo_onnx_postprocess import yolo_onnx_postprocess

        assert hasattr(yolo_onnx_postprocess, "__wrapped_function__")

    @pytest.mark.unit
    def test_no_detections_above_threshold(self):
        from tasks.onnx.yolo_onnx_postprocess import yolo_onnx_postprocess

        # Low-confidence outputs (all scores below threshold)
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        result = yolo_onnx_postprocess.__wrapped_function__(
            raw_output=raw_output,
            original_shape=(480, 640),
            confidence_threshold=0.5,
        )
        assert isinstance(result, list)
        assert result == []

    @pytest.mark.unit
    def test_list_input_handling(self):
        from tasks.onnx.yolo_onnx_postprocess import yolo_onnx_postprocess

        # Wrapping output in a list (as returned by onnx_inference)
        raw_output = [np.zeros((1, 84, 8400), dtype=np.float32)]
        result = yolo_onnx_postprocess.__wrapped_function__(
            raw_output=raw_output,
            original_shape=(480, 640),
        )
        assert isinstance(result, list)

    @pytest.mark.unit
    def test_empty_list_input(self):
        from tasks.onnx.yolo_onnx_postprocess import yolo_onnx_postprocess

        result = yolo_onnx_postprocess.__wrapped_function__(
            raw_output=[],
            original_shape=(480, 640),
        )
        assert result == []

