# FabricFlow Samples

This directory contains example code and media files demonstrating how to use FabricFlow.

## Directory Structure

- **api/** - REST API client examples showing how to interact with the FabricFlow service
- **sdk/** - Python SDK examples for local workflow execution
- **streamlit/** - Streamlit web applications for interactive workflow execution
- **media/** - Sample images and videos used by the workflow examples in `workflows/`

## Using Sample Workflows

The workflow definitions in `workflows/` reference sample media files using relative paths:

```json
{
  "inputs": {
    "image_path": {
      "literal": "samples/media/object_detection_sample.jpg"
    }
  }
}
```

## Adding Your Own Samples

1. Place your sample files in `samples/media/`
2. Update workflow definitions to reference them using relative paths from the repository root
3. The paths will be resolved automatically when running from the repository root via CLI or service

## Running Examples

See the README files in each subdirectory for specific usage instructions:
- [API Examples](api/README.md) - Using the REST API client
- [SDK Examples](sdk/README.md) - Using the Python SDK for local execution
- [Streamlit Apps](streamlit/README.md) - Interactive web interfaces for workflows
- [Media Files](media/README.md) - Sample images and videos for workflows
