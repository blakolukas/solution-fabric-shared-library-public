from core.task import task


@task(
    outputs=["extracted_text"],
    output_types={"extracted_text": "text"},
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
def extract_text_with_ocr(grayscale_image, language: str = "eng"):
    """
    Extract text from a preprocessed grayscale image using OCR.

    Args:
        grayscale_image: Preprocessed grayscale image
        language: Tesseract language code (default: "eng")

    Returns:
        Extracted text string
    """
    try:
        import pytesseract

        text = pytesseract.image_to_string(grayscale_image, lang=language)
        return text.strip()
    except Exception as e:
        raise RuntimeError(
            f"OCR text extraction failed. Ensure Tesseract is installed and configured: {e}"
        )
