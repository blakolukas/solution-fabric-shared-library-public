"""Smoke tests for tasks/workflow/ — verify importability and basic execution."""

import pytest


# ---------------------------------------------------------------------------
# get_workflow_context
# ---------------------------------------------------------------------------
class TestGetWorkflowContextSmoke:
    """Smoke tests for tasks.workflow.get_workflow_context."""

    def test_importable(self):
        from tasks.workflow.get_workflow_context import get_workflow_context

        assert hasattr(get_workflow_context, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.workflow.get_workflow_context import get_workflow_context

        context = {"key": "value", "count": 42}
        result = get_workflow_context.__wrapped_function__(context=context)
        assert result == context

    @pytest.mark.unit
    def test_empty_context(self):
        from tasks.workflow.get_workflow_context import get_workflow_context

        result = get_workflow_context.__wrapped_function__(context={})
        assert result == {}

    @pytest.mark.unit
    def test_passthrough_identity(self):
        from tasks.workflow.get_workflow_context import get_workflow_context

        ctx = {"nested": {"a": 1}, "list": [1, 2, 3]}
        result = get_workflow_context.__wrapped_function__(context=ctx)
        assert result is ctx
