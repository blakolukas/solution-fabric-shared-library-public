"""Smoke tests for tasks/vision/ — verify importability and basic execution."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# draw_detection_boxes
# ---------------------------------------------------------------------------
class TestDrawDetectionBoxesSmoke:
    """Smoke tests for tasks.vision.draw_detection_boxes."""

    def test_importable(self):
        from tasks.vision.draw_detection_boxes import draw_detection_boxes

        assert hasattr(draw_detection_boxes, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.vision.draw_detection_boxes import draw_detection_boxes

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            {
                "bbox": [10.0, 10.0, 50.0, 50.0],
                "class_name": "person",
                "confidence": 0.9,
                "class_id": 0,
            }
        ]
        result = draw_detection_boxes.__wrapped_function__(
            image=image, detections=detections
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape

    @pytest.mark.unit
    def test_no_detections(self):
        from tasks.vision.draw_detection_boxes import draw_detection_boxes

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = draw_detection_boxes.__wrapped_function__(image=image, detections=[])
        # Image should be returned unchanged (copy)
        assert result.shape == image.shape
        np.testing.assert_array_equal(result, image)


# ---------------------------------------------------------------------------
# load_yolo_model
# ---------------------------------------------------------------------------
class TestLoadYoloModelSmoke:
    """Smoke tests for tasks.vision.load_yolo_model."""

    def test_importable(self):
        from tasks.vision.load_yolo_model import load_yolo_model

        assert hasattr(load_yolo_model, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        import sys

        mock_yolo_instance = MagicMock()
        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_yolo_instance
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(
            sys.modules,
            {"torch": mock_torch, "ultralytics": mock_ultralytics},
        ):
            from tasks.vision.load_yolo_model import load_yolo_model

            result = load_yolo_model.__wrapped_function__(model_path="yolo11n.pt")

        assert result is mock_yolo_instance


# ---------------------------------------------------------------------------
# yolo_detect_objects
# ---------------------------------------------------------------------------
class TestYoloDetectObjectsSmoke:
    """Smoke tests for tasks.vision.yolo_detect_objects."""

    def test_importable(self):
        from tasks.vision.yolo_detect_objects import yolo_detect_objects

        assert hasattr(yolo_detect_objects, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.vision.yolo_detect_objects import yolo_detect_objects

        # Build mock result with a single detection
        mock_box = MagicMock()
        mock_box.xyxy.cpu.return_value.numpy.return_value = np.array([[10.0, 10.0, 50.0, 50.0]])
        mock_box.conf.cpu.return_value.numpy.return_value = np.array([0.85])
        mock_box.cls.cpu.return_value.numpy.return_value = np.array([0])
        mock_box.__len__ = lambda self: 1

        mock_result = MagicMock()
        mock_result.boxes = mock_box
        mock_result.names = {0: "person"}

        mock_model = MagicMock()
        mock_model.return_value = iter([mock_result])

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = yolo_detect_objects.__wrapped_function__(
            yolo_model=mock_model, image=image
        )
        assert isinstance(detections, list)
        assert len(detections) == 1
        assert detections[0]["class_name"] == "person"

    @pytest.mark.unit
    def test_no_boxes(self):
        from tasks.vision.yolo_detect_objects import yolo_detect_objects

        mock_box = MagicMock()
        mock_box.__len__ = lambda self: 0

        mock_result = MagicMock()
        mock_result.boxes = mock_box

        mock_model = MagicMock()
        mock_model.return_value = iter([mock_result])

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = yolo_detect_objects.__wrapped_function__(
            yolo_model=mock_model, image=image
        )
        assert detections == []


# ---------------------------------------------------------------------------
# blur_image_based_on_mask
# ---------------------------------------------------------------------------
class TestBlurImageBasedOnMaskSmoke:
    """Smoke tests for tasks.vision.blur_image_based_on_mask."""

    def test_importable(self):
        from tasks.vision.blur_image_based_on_mask import blur_image_with_mask

        assert hasattr(blur_image_with_mask, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.vision.blur_image_based_on_mask import blur_image_with_mask

        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)  # blur everything
        result = blur_image_with_mask.__wrapped_function__(
            image_array=image, mask_array=mask, blur_radius=5
        )
        assert result.shape == image.shape
        assert result.dtype == image.dtype

    @pytest.mark.unit
    def test_full_keep_mask(self):
        from tasks.vision.blur_image_based_on_mask import blur_image_with_mask

        image = np.ones((30, 30, 3), dtype=np.uint8) * 128
        mask = np.ones((30, 30), dtype=np.uint8) * 255  # keep everything
        result = blur_image_with_mask.__wrapped_function__(
            image_array=image, mask_array=mask, blur_radius=5
        )
        np.testing.assert_array_equal(result, image)


# ---------------------------------------------------------------------------
# invert_mask
# ---------------------------------------------------------------------------
class TestInvertMaskSmoke:
    """Smoke tests for tasks.vision.invert_mask."""

    def test_importable(self):
        from tasks.vision.invert_mask import invert_mask

        assert hasattr(invert_mask, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.vision.invert_mask import invert_mask

        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        result = invert_mask.__wrapped_function__(mask_array=mask)
        assert result[0, 0] == 255
        assert result[0, 1] == 0

    @pytest.mark.unit
    def test_invalid_input_raises(self):
        from tasks.vision.invert_mask import invert_mask

        with pytest.raises(ValueError):
            invert_mask.__wrapped_function__(mask_array="not an array")


# ---------------------------------------------------------------------------
# process_segmentation_mask
# ---------------------------------------------------------------------------
class TestProcessSegmentationMaskSmoke:
    """Smoke tests for tasks.vision.process_segmentation_mask."""

    def test_importable(self):
        from tasks.vision.process_segmentation_mask import process_segmentation_mask

        assert hasattr(process_segmentation_mask, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.vision.process_segmentation_mask import process_segmentation_mask

        # Mask with class_id 15 (person) and background (0)
        mask = np.array([[0, 15], [15, 0]], dtype=np.int32)
        result = process_segmentation_mask.__wrapped_function__(
            mask_array=mask, keep_class_id=15
        )
        assert result.shape == (2, 2)
        assert result[0, 1] == 255   # person → keep
        assert result[0, 0] == 0     # background → blur


# ---------------------------------------------------------------------------
# visualize_segmentation_mask
# ---------------------------------------------------------------------------
class TestVisualizeSegmentationMaskSmoke:
    """Smoke tests for tasks.vision.visualize_segmentation_mask."""

    def test_importable(self):
        from tasks.vision.visualize_segmentation_mask import visualize_segmentation_mask

        assert hasattr(visualize_segmentation_mask, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        import sys

        from tasks.vision.visualize_segmentation_mask import visualize_segmentation_mask

        mask = np.array([[0, 1], [1, 0]], dtype=np.int32)
        fake_colormap = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)
        mock_imgviz = MagicMock()
        mock_imgviz.label_colormap.return_value = fake_colormap

        with patch.dict(sys.modules, {"imgviz": mock_imgviz}):
            colored_mask, summary = visualize_segmentation_mask.__wrapped_function__(
                mask_array=mask
            )

        assert colored_mask.shape == (2, 2, 3)
        assert isinstance(summary, str)
        assert "class_0" in summary or "class_1" in summary


# ---------------------------------------------------------------------------
# export_model_to_onnx
# ---------------------------------------------------------------------------
class TestExportModelToOnnxSmoke:
    """Smoke tests for tasks.vision.export_model_to_onnx."""

    def test_importable(self):
        from tasks.vision.export_model_to_onnx import export_model_to_onnx

        assert hasattr(export_model_to_onnx, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self, tmp_path):
        from tasks.vision.export_model_to_onnx import export_model_to_onnx

        expected_path = str(tmp_path / "model.onnx")
        mock_model = MagicMock()
        mock_model.export.return_value = expected_path

        result = export_model_to_onnx.__wrapped_function__(
            yolo_model=mock_model, imgsz=640, simplify=True
        )
        assert result == expected_path
        mock_model.export.assert_called_once_with(
            format="onnx", imgsz=640, simplify=True, dynamic=False, opset=17
        )
