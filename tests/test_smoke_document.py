"""Smoke tests for tasks/document/ — verify importability and basic execution."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# load_pdf
# ---------------------------------------------------------------------------
class TestLoadPdfSmoke:
    """Smoke tests for tasks.document.load_pdf."""

    def test_importable(self):
        from tasks.document.load_pdf import load_pdf

        assert hasattr(load_pdf, "__wrapped_function__")

    @pytest.mark.unit
    def test_returns_path(self, tmp_path):
        """load_pdf validates extension and returns the path unchanged."""
        from tasks.document.load_pdf import load_pdf

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")  # minimal PDF-like content
        result = load_pdf.__wrapped_function__(str(pdf_file))
        assert result == str(pdf_file)

    @pytest.mark.unit
    def test_wrong_extension_raises(self, tmp_path):
        from tasks.document.load_pdf import load_pdf

        non_pdf = tmp_path / "test.txt"
        non_pdf.write_text("hello")
        with pytest.raises(ValueError, match="Expected PDF"):
            load_pdf.__wrapped_function__(str(non_pdf))

    @pytest.mark.unit
    def test_empty_input_raises(self):
        from tasks.document.load_pdf import load_pdf

        with pytest.raises(ValueError):
            load_pdf.__wrapped_function__("")


# ---------------------------------------------------------------------------
# extract_text_with_ocr
# ---------------------------------------------------------------------------
class TestExtractTextWithOcrSmoke:
    """Smoke tests for tasks.document.extract_text_with_ocr."""

    def test_importable(self):
        from tasks.document.extract_text_with_ocr import extract_text_with_ocr

        assert hasattr(extract_text_with_ocr, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        from tasks.document.extract_text_with_ocr import extract_text_with_ocr

        mock_pytesseract = MagicMock()
        mock_pytesseract.image_to_string.return_value = "  Hello World  "
        grayscale_img = np.zeros((100, 100), dtype=np.uint8)

        with patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            result = extract_text_with_ocr.__wrapped_function__(grayscale_img)

        assert result == "Hello World"


# ---------------------------------------------------------------------------
# extract_ocr_data
# ---------------------------------------------------------------------------
class TestExtractOcrDataSmoke:
    """Smoke tests for tasks.document.extract_ocr_data."""

    def test_importable(self):
        from tasks.document.extract_ocr_data import extract_ocr_data

        assert hasattr(extract_ocr_data, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        from tasks.document.extract_ocr_data import extract_ocr_data

        expected_data = {
            "text": ["Hello", "World"],
            "left": [10, 20],
            "top": [5, 5],
            "height": [15, 15],
            "conf": [90, 85],
        }
        mock_pytesseract = MagicMock()
        mock_pytesseract.image_to_data.return_value = expected_data
        mock_pytesseract.Output = MagicMock()
        mock_pytesseract.Output.DICT = "dict"

        grayscale_img = np.zeros((100, 100), dtype=np.uint8)

        with patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            result = extract_ocr_data.__wrapped_function__(grayscale_img)

        assert result == expected_data


# ---------------------------------------------------------------------------
# create_searchable_pdf
# ---------------------------------------------------------------------------
class TestCreateSearchablePdfSmoke:
    """Smoke tests for tasks.document.create_searchable_pdf."""

    def test_importable(self):
        from tasks.document.create_searchable_pdf import create_searchable_pdf

        assert hasattr(create_searchable_pdf, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self, tmp_path):
        from tasks.document.create_searchable_pdf import create_searchable_pdf

        # Minimal OCR data with a single text entry
        ocr_data = {
            "text": ["Hello"],
            "left": [10],
            "top": [10],
            "height": [20],
            "conf": [90],
        }
        image_dimensions = (200, 300)  # (height, width)
        image_path = str(tmp_path / "img.png")

        # Build a mock that mirrors `from reportlab.pdfgen import canvas; canvas.Canvas(...)`
        mock_canvas_instance = MagicMock()
        mock_canvas_cls = MagicMock(return_value=mock_canvas_instance)

        mock_canvas_submodule = MagicMock()
        mock_canvas_submodule.Canvas = mock_canvas_cls

        mock_pdfgen = MagicMock()
        mock_pdfgen.canvas = mock_canvas_submodule

        mock_color_instance = MagicMock()
        mock_colors = MagicMock()
        mock_colors.Color.return_value = mock_color_instance

        mock_reportlab_lib = MagicMock()
        mock_reportlab_lib.colors = mock_colors

        mock_reportlab = MagicMock()
        mock_reportlab.lib = mock_reportlab_lib
        mock_reportlab.pdfgen = mock_pdfgen

        with patch.dict(
            sys.modules,
            {
                "reportlab": mock_reportlab,
                "reportlab.lib": mock_reportlab_lib,
                "reportlab.lib.colors": mock_colors,
                "reportlab.pdfgen": mock_pdfgen,
                "reportlab.pdfgen.canvas": mock_canvas_submodule,
            },
        ):
            result = create_searchable_pdf.__wrapped_function__(
                image_path=image_path,
                ocr_data=ocr_data,
                image_dimensions=image_dimensions,
                output_directory=str(tmp_path),
                output_name="output.pdf",
            )

        import os

        assert result == os.path.join(str(tmp_path), "output.pdf")
        mock_canvas_cls.assert_called_once()
        mock_canvas_instance.save.assert_called_once()
