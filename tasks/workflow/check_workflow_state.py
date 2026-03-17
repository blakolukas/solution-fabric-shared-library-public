"""Get the state of a workflow on a FabricFlow agent."""

import requests
from core.task import task


@task(
    outputs=["state", "instance_count"],
    output_types={"state": "str", "instance_count": "int"},
    display_name="Get Workflow State",
    description="Get the current state of a workflow on a FabricFlow agent",
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
    Query the agent for instances of the workflow and return their state.

    Returns:
        state: 'idle', 'running', 'error', or 'not_loaded' if no instances exist
        instance_count: number of instances found
    """
    try:
        response = requests.get(
            f"{agent_url}/instances",
            params={"workflow_name": workflow_name},
            timeout=5,
        )
        if response.status_code == 200:
            data = response.json()
            instances = data.get("instances", [])
            if not instances:
                return "not_loaded", 0
            state = instances[0].get("status", "unknown")
            return state, len(instances)
        return "unknown", 0
    except requests.RequestException:
        return "unreachable", 0