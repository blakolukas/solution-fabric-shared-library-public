import os
import zipfile

from core.task import task


@task(
    outputs=["onnx_model_path"],
    parameters={
        "model_path": {
            "type": "str",
            "required": True,
            "description": "Path to the downloaded model zip file",
        },
        "repository_id": {
            "type": "str",
            "required": True,
            "description": "Repository ID to determine extraction directory",
        },
        "base_directory": {
            "type": "str",
            "required": False,
            "default": "models",
            "description": "Base directory for model storage",
        },
    },
)
def extract_onnx_model(model_path: str, repository_id: str, base_directory: str = "models"):
    """
    Extract ONNX model from a zip file.

    Args:
        model_path: Path to the downloaded model zip file
        repository_id: Repository ID to determine extraction directory
        base_directory: Base directory for model storage (default: "models")

    Returns:
        Path to extracted ONNX model
    """
    model_name = repository_id.split("/")[-1]
    model_directory = os.path.join(base_directory, model_name)

    # Extract the zip file
    existing_model_path = None
    if os.path.exists(model_directory):
        # Check flat structure first
        direct_path = os.path.join(model_directory, "model.onnx")
        if os.path.isfile(direct_path):
            existing_model_path = direct_path
        else:
            # Check nested structure
            nested_path = os.path.join(model_directory, "model.onnx", "model.onnx")
            if os.path.isfile(nested_path):
                existing_model_path = nested_path
            else:
                # Search recursively as fallback
                for root, dirs, files in os.walk(model_directory):
                    if "model.onnx" in files:
                        existing_model_path = os.path.join(root, "model.onnx")
                        break

    if existing_model_path:
        return existing_model_path

    os.makedirs(model_directory, exist_ok=True)

    with zipfile.ZipFile(model_path, "r") as zip_ref:
        zip_ref.extractall(model_directory)

    actual_model_path = None
    direct_path = os.path.join(model_directory, "model.onnx")
    if os.path.isfile(direct_path):
        actual_model_path = direct_path
    else:
        # Check for nested structure (model.onnx/model.onnx)
        nested_path = os.path.join(model_directory, "model.onnx", "model.onnx")
        if os.path.isfile(nested_path):
            actual_model_path = nested_path
        else:
            # Search recursively as fallback
            for root, dirs, files in os.walk(model_directory):
                if "model.onnx" in files:
                    actual_model_path = os.path.join(root, "model.onnx")
                    break

    if not actual_model_path:
        raise FileNotFoundError(f"model.onnx not found in extracted contents at {model_directory}")

    return actual_model_path
