"""Read current CPU usage from the host machine."""

import psutil

from core.task import task


@task(
    outputs=["cpu_percent", "core_count", "frequency_mhz"],
    display_name="Get CPU Usage",
    description="Read current CPU utilization, core count, and clock frequency from the host machine",
    category="system",
    output_types={
        "cpu_percent": "float",
        "core_count": "int",
        "frequency_mhz": "float",
    },
    is_collapsed=True,
    parameters={},
)
def get_cpu_usage():
    """
    Read current CPU usage from the host machine.

    Uses a 0.1-second blocking interval for an accurate instantaneous
    utilization sample rather than the non-blocking (always-0) first call.

    Returns:
        cpu_percent: Overall CPU usage percentage (0–100)
        core_count: Number of logical CPU cores
        frequency_mhz: Current CPU clock frequency in MHz (0.0 if unavailable)
    """
    cpu_percent = psutil.cpu_percent(interval=0.1)
    core_count = psutil.cpu_count(logical=True) or 0

    freq = psutil.cpu_freq()
    frequency_mhz = round(freq.current, 2) if freq else 0.0

    return cpu_percent, core_count, frequency_mhz
