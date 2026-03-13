import os

from core.task import task


@task(
    outputs=["login_status"],
    output_types={"login_status": "bool"},
    parameters={
        "token": {
            "type": "str",
            "required": False,
            "default": None,
            "description": "HuggingFace API token",
        },
        "add_to_git_credential": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Whether to add token to git credential store",
        },
        "new_session": {
            "type": "bool",
            "required": False,
            "default": True,
            "description": "Whether to create a new session",
        },
    },
    is_collapsed=True,
)
def huggingface_login(
    token: str = None, add_to_git_credential: bool = False, new_session: bool = True
):
    """Login to HuggingFace Hub to access gated models and datasets.

    Args:
        token: HuggingFace API token. If None, will check HF_TOKEN env variable
        add_to_git_credential: Whether to add token to git credential store
        new_session: Whether to create a new session

    Returns:
        login_status: Status message indicating successful login
    """
    from huggingface_hub import login

    # Check for token in environment variable if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError(
                "No HuggingFace token provided. "
                "Please set HF_TOKEN environment variable or pass token parameter."
            )

    # Login to HuggingFace Hub
    print("Logging in to HuggingFace Hub...")
    login(
        token=token,
        add_to_git_credential=add_to_git_credential,
        new_session=new_session,
    )

    print("Successfully logged in to HuggingFace Hub")
    return "logged_in"
