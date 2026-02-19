from core.task import task


@task(
    outputs=["dataset"],
    parameters={
        "dataset_name": {
            "type": "str",
            "required": False,
            "default": None,
            "description": "HuggingFace dataset name",
        },
        "dataset_path": {
            "type": "str",
            "required": False,
            "default": None,
            "description": "Local path to dataset files",
        },
        "split": {
            "type": "str",
            "required": False,
            "default": "train",
            "description": "Dataset split to load",
        },
        "streaming": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Whether to stream the dataset",
        },
        "trust_remote_code": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Trust remote code from HuggingFace",
        },
    },
)
def load_training_dataset(
    dataset_name: str = None,
    dataset_path: str = None,
    split: str = "train",
    streaming: bool = False,
    trust_remote_code: bool = False,
):
    """Load training dataset from HuggingFace Hub or local path.

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'squad', 'coco')
        dataset_path: Local path to dataset files
        split: Dataset split to load ('train', 'validation', 'test')
        streaming: Whether to stream the dataset instead of downloading
        trust_remote_code: Whether to trust remote code (needed for some datasets)
    """
    from datasets import load_dataset
    
    if dataset_name:
        dataset = load_dataset(dataset_name, split=split, streaming=streaming, trust_remote_code=trust_remote_code)
    elif dataset_path:
        dataset = load_dataset(dataset_path, split=split, streaming=streaming, trust_remote_code=trust_remote_code)
    else:
        raise ValueError("Either dataset_name or dataset_path must be provided")

    return dataset
