# Sample Media Files

This directory contains sample media files used by the workflow examples in `workflows/`.

## Files

- **object_detection_sample.jpg** - Sample image for YOLO object detection workflow
- **document_sample.jpg** - Sample document image for OCR/document scanning workflow  
- **video_stream_sample.mp4** - Sample video for streaming video detection workflows

## Usage

The workflow JSON files reference these samples using relative paths like:
```json
"image_path": {
  "literal": "samples/media/object_detection_sample.jpg"
}
```

This ensures workflows work hermetically across different environments without requiring absolute paths specific to one user's file system.

## Adding Your Own Samples

Place your sample media files in this directory and update the workflow definitions to reference them using the relative path pattern shown above.
