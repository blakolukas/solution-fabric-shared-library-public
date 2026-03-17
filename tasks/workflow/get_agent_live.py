"""Check if a FabricFlow agent is live and healthy."""

import requests
from core.task import task


@task(
    outputs=["is_live", "agent_info"],
    output_types={"is_live": "bool", "agent_info": "dict"},
    display_name="Get Agent Live",
    description="Get if a FabricFlow agent is live and return its info",
    category="workflow",
    parameters={
        "agent_url": {
            "type": "str",
            "required": True,
            "description": "Base URL of the FabricFlow agent (e.g. http://192.168.1.10:8000)",
        },
    },
)
def get_agent_live(agent_url: str) -> tuple:
    """
    Hit the agent /health endpoint and return liveness + agent metadata.

    Returns:
        is_live: True if the agent responded with status 200
        agent_info: Dict with name, agent_id, hostname, platform (empty if not live)
    """
    try:
        response = requests.get(f"{agent_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data.get("agent", {})
        return False, {}
    except requests.RequestException:
        return False, {}