from core.task import task


@task(
    outputs=["model_path"],
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
    },
)
def download_hf_model(repository_id: str, model_name: str):
    """
    Download a model from HuggingFace Hub.

    Args:
        repository_id: HuggingFace repository ID (e.g., 'username/model-name')
        model_name: Name of the model file to download

    Returns:
        Path to the cached file
    """
    from huggingface_hub import hf_hub_download
    
    return hf_hub_download(repo_id=repository_id, filename=model_name)
