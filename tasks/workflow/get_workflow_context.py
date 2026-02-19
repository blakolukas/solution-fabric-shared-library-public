from core.task import task


@task(
    outputs=["context"],
    parameters={
        "context": {
            "type": "dict",
            "required": True,
            "description": "Current workflow context",
        },
    },
)
def get_workflow_context(context: dict):
    """
    Return the current workflow context.

    Returns:
        The current workflow context
    """
    return context
