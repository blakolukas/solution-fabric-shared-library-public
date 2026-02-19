"""
Streamlit Workflow Client for FabricFlow Applications

Provides a simplified interface to the FabricFlow API with
error handling and retry logic suitable for interactive UIs.

Note: This is distinct from utils/fabricflow_client.py which is a
low-level HTTP client with binary data handling and connection pooling.
This wrapper uses the official fabricflow.Client SDK and adds
Streamlit-specific UI features (progress bars, error messages).
"""

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional

import streamlit as st
from fabricflow import Client
from fabricflow.api.client import APIError, ServerConnectionError


class StreamlitWorkflowClient:
    """
    Streamlit-friendly wrapper around fabricflow.Client.

    Provides workflow execution with progress bars and user-friendly error handling.
    For low-level API access with binary data support, use utils/fabricflow_client.py instead.

    Example:
        client = StreamlitWorkflowClient("http://localhost:8000")
        result = client.execute_workflow_sync(
            "my_workflow",
            inputs={"text": "Hello"},
            progress_callback=lambda msg: st.text(msg)
        )
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        """
        Initialize the FabricFlow client.

        Args:
            base_url: URL of the FabricFlow server
        """
        self.base_url = base_url
        self._client = None

    @property
    def client(self) -> Client:
        """Lazy initialization of the underlying client."""
        if self._client is None:
            self._client = Client(self.base_url)
        return self._client

    def test_connection(self) -> tuple[bool, str]:
        """
        Test connection to the FabricFlow server.

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            health = self.client.health()
            status = health.get("status", "unknown") if isinstance(health, dict) else "ok"
            return True, f"Connected: {status}"
        except ServerConnectionError as e:
            return False, f"Cannot connect: {str(e)}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def list_workflows(self) -> list[Dict[str, Any]]:
        """
        List all available workflows.

        Returns:
            list: Workflow definitions
        """
        try:
            response = self.client.workflow.list()
            if isinstance(response, dict):
                return response.get("workflows", [])
            elif isinstance(response, list):
                return response
            return []
        except Exception as e:
            st.error(f"❌ Failed to list workflows: {str(e)}")
            return []

    def workflow_exists(self, workflow_name: str) -> bool:
        """
        Check if a workflow exists.

        Args:
            workflow_name: Name of the workflow

        Returns:
            bool: True if workflow exists
        """
        workflows = self.list_workflows()
        for w in workflows:
            if isinstance(w, dict):
                if w.get("name") == workflow_name:
                    return True
            elif isinstance(w, str):
                if w == workflow_name:
                    return True
        return False

    @contextmanager
    def _workflow_instance(self, workflow_name: str, timeout: int) -> Generator[str, None, None]:
        """
        Context manager for workflow instance lifecycle.

        Handles instantiation and cleanup of workflow instances.

        Args:
            workflow_name: Name of the workflow
            timeout: TTL for the instance in seconds

        Yields:
            str: The instance GUID
        """
        instance_guid = None
        try:
            instance = self.client.workflow.instantiate(workflow_name, ttl_seconds=max(timeout, 3600))
            instance_guid = instance["instance_guid"]
            yield instance_guid
        finally:
            if instance_guid:
                try:
                    self.client.instance.terminate(instance_guid)
                except Exception:
                    pass  # Ignore cleanup errors

    def execute_workflow_sync(
        self,
        workflow_name: str,
        inputs: Dict[str, Any],
        progress_callback: Optional[Callable[[str], None]] = None,
        timeout: int = 300,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a workflow synchronously with progress updates.

        Args:
            workflow_name: Name of the workflow to execute
            inputs: Input parameters for the workflow
            progress_callback: Optional callback for progress updates
            timeout: Maximum wait time in seconds

        Returns:
            dict: Workflow results or None on failure
        """
        try:
            # Validate workflow exists
            if not self.workflow_exists(workflow_name):
                st.error(f"❌ Workflow '{workflow_name}' not found")
                return None

            with self._workflow_instance(workflow_name, timeout) as instance_guid:
                # Execute workflow in sync mode
                result = self.client.instance.execute(instance_guid, inputs=inputs, mode="sync", output_format="json")
                return result

        except ServerConnectionError as e:
            st.error(f"❌ Connection failed: {str(e)}")
            st.info("💡 Ensure FabricFlow server is running")
            return None

        except APIError as e:
            st.error(f"❌ Workflow execution failed: {str(e)}")
            return None

        except Exception as e:
            handle_api_error(e)
            return None

    def execute_workflow_with_progress_bar(
        self, workflow_name: str, inputs: Dict[str, Any], timeout: int = 300
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a workflow with Streamlit progress bar and status text.
        Uses sync mode for simpler execution flow.

        Args:
            workflow_name: Name of the workflow to execute
            inputs: Input parameters for the workflow
            timeout: Maximum wait time in seconds

        Returns:
            dict: Workflow results or None on failure
        """
        progress_bar = st.progress(0, text="Initializing...")
        status_text = st.empty()

        try:
            # Validate workflow exists
            if not self.workflow_exists(workflow_name):
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Workflow '{workflow_name}' not found")
                return None

            # Instantiate
            status_text.text("Creating workflow instance...")
            progress_bar.progress(20)

            with self._workflow_instance(workflow_name, timeout) as instance_guid:
                # Execute in sync mode
                status_text.text("Executing workflow...")
                progress_bar.progress(50)

                # Sync mode blocks until completion
                result = self.client.instance.execute(instance_guid, inputs=inputs, mode="sync", output_format="json")

                # Success
                progress_bar.progress(100)
                status_text.success("✅ Completed!")

                time.sleep(0.5)  # Brief pause to show completion
                progress_bar.empty()
                status_text.empty()

                # IMPORTANT: Extract outputs from nested response
                # API returns: {"execution_id": "...", "outputs": {...}, "status": "..."}
                # Apps expect workflow results directly
                if isinstance(result, dict) and "outputs" in result:
                    return result["outputs"]
                return result

        except ServerConnectionError as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Failed to connect to FabricFlow server: {str(e)}")
            st.info("💡 Please ensure the server is running on the configured URL")
            return None

        except APIError as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Workflow execution failed: {str(e)}")
            return None

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            handle_api_error(e)
            return None

    def close(self):
        """Close the underlying client connection."""
        if self._client:
            self._client.close()


def handle_api_error(error: Exception):
    """
    Display user-friendly error messages for common API errors.

    Args:
        error: The exception to handle
    """
    import requests.exceptions

    if isinstance(error, requests.exceptions.ConnectionError):
        st.error("❌ Cannot connect to FabricFlow server")
        st.info("💡 Make sure the server is running: `fabric run`")

    elif isinstance(error, requests.exceptions.Timeout):
        st.error("❌ Request timed out")
        st.info("💡 Try increasing the timeout or check server load")

    elif isinstance(error, requests.exceptions.HTTPError):
        status_code = error.response.status_code

        if status_code == 404:
            st.error("❌ Resource not found")
            st.info("💡 Check that the workflow name is correct")

        elif status_code == 500:
            st.error("❌ Server error")
            try:
                error_detail = error.response.json().get("detail", str(error))
                st.error(f"Details: {error_detail}")
            except Exception:
                st.error(f"Details: {error.response.text}")

        elif status_code == 422:
            st.error("❌ Invalid request parameters")
            try:
                error_detail = error.response.json()
                st.json(error_detail)
            except Exception:
                st.error(f"Details: {error.response.text}")

        else:
            st.error(f"❌ HTTP error {status_code}: {str(error)}")

    else:
        st.error(f"❌ Unexpected error: {str(error)}")

        # Show full traceback in expander
        with st.expander("📋 Full Error Details"):
            st.exception(error)


def format_execution_time(seconds: float) -> str:
    """
    Format execution time in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        str: Formatted time string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
