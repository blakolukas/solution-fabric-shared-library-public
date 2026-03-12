"""Return the current date and time as an ISO 8601 timestamp."""

from datetime import datetime

from core.task import task


@task(
    outputs=["timestamp"],
    display_name="Get Current Timestamp",
    description="Return the current local date and time as an ISO 8601 string",
    category="system",
    output_types={"timestamp": "str"},
    is_collapsed=True,
    parameters={},
)
def get_current_timestamp():
    """
    Return the current date and time as an ISO 8601 timestamp.

    Uses the local system clock. No external dependencies.

    Returns:
        timestamp: Current datetime string, e.g. '2026-03-12T14:35:07.123456'
    """
    return datetime.now().isoformat()
