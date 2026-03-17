"""Check if a workflow is deployed on a FabricFlow agent."""

import requests
from core.task import task


@task(
    outputs=["is_deployed"],
    output_types={"is_deployed": "bool"},
    display_name="Get Workflow State",
    description="Check if a workflow is deployed on a FabricFlow agent",
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
def get_workflow_state(agent_url: str, workflow_name: str) -> bool:
    """
    Query the agent for the workflow definition and return whether it exists.

    Returns:
        is_deployed: True if the workflow exists on the agent, False otherwise
    """
    try:
        response = requests.get(
            f"{agent_url}/workflows/{workflow_name}",
            timeout=5,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False