from core.task import task


@task(
    outputs=["pdf"],
    output_types={"pdf": "pdf"},
    display_name="Create Searchable PDF",
    description="Create a searchable PDF from an image with OCR text overlay",
    category="document",
    parameters={
        "image_path": {
            "type": "str",
            "required": True,
            "description": "Path to the document image",
        },
        "ocr_data": {
            "type": "dict",
            "required": True,
            "description": "OCR data with text positions and bounding boxes",
        },
        "image_dimensions": {
            "type": "tuple",
            "required": True,
            "description": "Tuple of (height, width) of the image",
        },
        "output_directory": {
            "type": "str",
            "required": True,
            "description": "Directory to save PDF",
        },
        "output_name": {
            "type": "str",
            "required": False,
            "default": "output.pdf",
            "description": "PDF filename",
        },
    },
)
def create_searchable_pdf(
    image_path: str, ocr_data: dict, image_dimensions: tuple, output_directory: str, output_name: str = "output.pdf"
):
    """
    Create and save a searchable PDF that mirrors the original document.

    Creates a PDF with the original image as background and invisible
    text overlay positioned to match the OCR locations, making the
    PDF a searchable digital twin of the scanned document.

    Args:
        image_path: Path to the document image
        ocr_data: OCR data with text positions and bounding boxes
        image_dimensions: Tuple of (height, width) of the image
        output_directory: Directory to save PDF (must exist)
        output_name: PDF filename (default: "output.pdf")

    Returns:
        Full path to the generated PDF
    """
    import os

    from reportlab.lib.colors import Color
    from reportlab.pdfgen import canvas

    full_pdf_path = os.path.join(output_directory, output_name)

    img_height, img_width = image_dimensions
    page_width = img_width
    page_height = img_height

    c = canvas.Canvas(full_pdf_path, pagesize=(page_width, page_height))

    c.drawImage(image_path, 0, 0, width=page_width, height=page_height)

    transparent = Color(1, 1, 1, alpha=0)
    c.setFillColor(transparent)

    texts = ocr_data.get("text", [])
    lefts = ocr_data.get("left", [])
    tops = ocr_data.get("top", [])
    heights = ocr_data.get("height", [])
    confs = ocr_data.get("conf", [])

    for i, text in enumerate(texts):
        if text.strip() and int(confs[i]) > 0:
            x = lefts[i]
            y = img_height - tops[i] - heights[i]
            font_size = max(heights[i] * 0.8, 6)

            c.setFont("Helvetica", font_size)
            c.drawString(x, y, text)

    c.save()

    return full_pdf_path
