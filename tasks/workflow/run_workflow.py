from typing import Any, Dict, Optional

import requests

from requests import HTTPError

from core.task import task

 

def _build_headers(session_guid: Optional[str] = None) -> dict:

    if session_guid:

        return {"X-Session-GUID": session_guid}

    return {}

 

def _create_session(base_url: str) -> str:

    """Create a new session and return the session GUID."""

    response = requests.post(

        f"{base_url}/sessions/create",

        json={"ttl_hours": 24, "metadata": {"client": "run_workflow_task"}},

        timeout=10,

    )

    response.raise_for_status()

    return response.json()["session_guid"]

 

def _create_instance(base_url: str, workflow_name: str, session_guid: str, ttl_seconds: int, instantiate: bool = True) -> Optional[str]:

    """Instantiate a workflow and return the instance GUID, or None if not instantiating."""

    if not instantiate:

        return None

    response = requests.post(

        f"{base_url}/workflows/{workflow_name}/instantiate",

        json={"ttl_seconds": ttl_seconds, "metadata": {"created_by": "run_workflow_task"}},

        headers=_build_headers(session_guid),

        timeout=10,

    )

    response.raise_for_status()

    return response.json()["instance_guid"]

 

def _is_workflow_not_found(error: HTTPError) -> bool:

    response = error.response

    if response is None or response.status_code != 404:

        return False

    try:

        payload = response.json()

    except ValueError:

        payload = None

 

    # Keep this broad because different agent versions return different 404 payloads.

    body = str(payload if payload is not None else response.text).lower()

    return "workflow" in body and ("not found" in body or "does not exist" in body)

 

def _execute_instance(

    base_url: str,

    instance_guid: str,

    params: Dict[str, Any],

    execution_mode: str,

    session_guid: Optional[str],

) -> dict:

    """Execute a workflow instance and return the raw response dict."""

    response = requests.post(

        f"{base_url}/instances/{instance_guid}/execute",

        json={

            "inputs": params,

            "execution_mode": execution_mode,

            "output_format": "json",

        },

        headers=_build_headers(session_guid),

        timeout=300,

    )

    response.raise_for_status()

    return response.json()

 

@task(

    outputs=["execution_id", "status", "outputs", "instance_guid"],

    display_name="Run Workflow",

    description="Runs any workflow on a FabricFlow agent, using an existing instance or creating a new one.",

    category="workflow",

    output_types={

        "execution_id": "str",

        "status": "str",

        "outputs": "dict",

        "instance_guid": "str",

    },

 

    parameters={

        "workflow_name": {

            "type": "str",

            "required": True,

            "description": "Name of the workflow to run.",

        },

 

        "params": {

            "type": "dict",

            "required": False,

            "default": {},

            "description": "Mutable input parameter dict forwarded to the workflow.Leave empty if the workflow require no inputs.",

 

        },

 

        "agent_id": {

            "type": "str",

            "required": True,

            "description": "Base URL of the FabricFlow agent to connect to (e.g. http://127.0.0.1:8000).",

        },

 

        "use_existing_instance": {

            "type": "bool",

            "required": False,

            "default": False,

            "description": "If True, reuse the instance identified by instance_guid instead of creating a new one.",

        },

 

        "instance_guid": {

            "type": "str",

            "required": False,

            "default": None,

            "description": "GUID of an existing instance to reuse. Required when use_existing_instance is True.",

        },

 

        "execution_mode": {

            "type": "str",

            "required": False,

            "default": "sync",

            "description": "Execution mode: 'sync' (wait for result) or 'async' (return immediately).",

        },

 

        "ttl_seconds": {

            "type": "int",

            "required": False,

            "default": 3600,

            "description": "Time-to-live for newly created instances, in seconds. Ignored when reusing an existing instance.",

        },

 

        "session_guid": {

            "type": "str",

            "required": False,

            "default": None,

            "description": "Existing session GUID to attach the instance to. A new session is created if not provided.",

        },

    },

)

 

def run_workflow(

    workflow_name: str,

    agent_id: str,

    params: Dict[str, Any] = {},

    use_existing_instance: bool = False,

    instance_guid: Optional[str] = None,

    execution_mode: str = "sync",

    ttl_seconds: int = 3600,

    session_guid: Optional[str] = None,

    instantiate: bool = True,

):

    base_url = agent_id.rstrip("/")

    active_session_guid = session_guid

 

    if use_existing_instance:

        if not instance_guid:

            raise ValueError("instance_guid must be provided when use_existing_instance is True.")

        active_instance_guid = instance_guid

    else:

        if not active_session_guid:

            active_session_guid = _create_session(base_url)

        try:

            active_instance_guid = _create_instance(base_url, workflow_name, active_session_guid, ttl_seconds, instantiate=instantiate)

 

        except HTTPError as error:

            if _is_workflow_not_found(error):

                return (

                    None,

                    "not_exists",

                    {

                        "message": f'Workflow "{workflow_name}" does not exist on the agent.',

                        "workflow_name": workflow_name,

                    },

                    None,

                )

            raise

    result = _execute_instance(base_url, active_instance_guid, params, execution_mode, active_session_guid)

 

    return (

        result.get("execution_id", ""),

        "exists" if result.get("status") in ("completed", "success") else result.get("status", ""),

        result.get("outputs") or {},

        active_instance_guid,

    )
