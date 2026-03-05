from core.task import task


@task(
    outputs=["ocr_data"],
    parameters={
        "grayscale_image": {
            "type": "array",
            "required": True,
            "description": "Preprocessed grayscale image",
        },
        "language": {
            "type": "str",
            "required": False,
            "default": "eng",
            "description": "Tesseract language code",
        },
    },
)
def extract_ocr_data(grayscale_image, language: str = "eng"):
    """
    Extract detailed OCR data including character positions and bounding boxes.

    Returns structured data with text, positions, and confidence scores
    that can be used to create a digital twin PDF.

    Args:
        grayscale_image: Preprocessed grayscale image
        language: Tesseract language code (default: "eng")

    Returns:
        Dictionary with OCR data including text, boxes, confidence levels
    """
    try:
        import pytesseract

        data = pytesseract.image_to_data(grayscale_image, lang=language, output_type=pytesseract.Output.DICT)
        return data
    except Exception as e:
        raise RuntimeError(f"OCR data extraction failed. Ensure Tesseract is installed and configured: {e}")
