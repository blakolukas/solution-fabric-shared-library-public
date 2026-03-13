from core.task import task


@task(
    outputs=["model_path"],
    output_types={"model_path": "str"},
    parameters={
        "repository_id": {
            "type": "str",
            "required": True,
            "description": "HuggingFace repository ID",
        },
        "model_name": {
            "type": "str",
            "required": True,
            "description": "Name of the model file to download",
        },
        "revision": {
            "type": "str",
            "required": False,
            "default": None,
            "description": "Branch, tag, or commit hash to download from",
        },
    },
    is_collapsed=True,
)
def download_hf_model(repository_id: str, model_name: str, revision: str = None):
    """
    Download a model from HuggingFace Hub.

    Args:
        repository_id: HuggingFace repository ID (e.g., 'username/model-name')
        model_name: Name of the model file to download

    Returns:
        Path to the cached file
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repository_id, filename=model_name, revision=revision
    )
