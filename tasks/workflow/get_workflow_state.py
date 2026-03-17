"""Get the deployment state of a workflow on a FabricFlow agent."""

import requests
from core.task import task


@task(
    outputs=["state", "is_deployed"],
    output_types={"state": "str", "is_deployed": "bool"},
    display_name="Get Workflow State",
    description="Get the deployment state of a workflow on a FabricFlow agent",
    category="workflow",
    parameters={
        "agent_url": {
            "type": "str",
            "required": True,
            "description": "Base URL of the FabricFlow agent",
        },
        "workflow_name": {
            "type": "str",
            "required": True,
            "description": "Name of the workflow to check",
        },
    },
)
def get_workflow_state(agent_url: str, workflow_name: str) -> tuple:
    """
    Query the agent for the workflow definition and return its deployment state.

    Returns:
        state: 'deployed', 'not_deployed', or 'unreachable'
        is_deployed: True if the workflow exists on the agent
    """
    try:
        response = requests.get(
            f"{agent_url}/workflows/{workflow_name}",
            timeout=5,
        )
        if response.status_code == 200:
            return "deployed", True
        if response.status_code == 404:
            return "not_deployed", False
        return "unknown", False
    except requests.RequestException:
        return "unreachable", False