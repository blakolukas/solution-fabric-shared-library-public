"""Read current disk usage for a given path on the host machine."""

import psutil

from core.task import task


@task(
    outputs=["disk_percent", "disk_used_gb", "disk_total_gb", "disk_free_gb"],
    display_name="Get Disk Usage",
    description="Read disk utilization and capacity for a given mount path on the host machine",
    category="system",
    output_types={
        "disk_percent": "float",
        "disk_used_gb": "float",
        "disk_total_gb": "float",
        "disk_free_gb": "float",
    },
    is_collapsed=True,
    parameters={
        "path": {
            "type": "str",
            "required": False,
            "default": "/",
            "description": "Mount path to inspect (e.g. '/' on Linux/macOS, 'C:\\' on Windows)",
        },
    },
)
def get_disk_usage(path: str = "/"):
    """
    Read current disk usage for a given path on the host machine.

    Args:
        path: Mount path to inspect. Defaults to '/' (root). Use 'C:\\' on Windows.

    Returns:
        disk_percent: Disk usage percentage (0–100)
        disk_used_gb: Used disk space in gigabytes
        disk_total_gb: Total disk capacity in gigabytes
        disk_free_gb: Free disk space in gigabytes
    """
    import os

    # On Windows default "/" resolves to the drive of the current working directory
    resolved_path = os.path.expanduser(path)

    usage = psutil.disk_usage(resolved_path)

    _GB = 1024**3
    disk_percent = usage.percent
    disk_used_gb = round(usage.used / _GB, 2)
    disk_total_gb = round(usage.total / _GB, 2)
    disk_free_gb = round(usage.free / _GB, 2)

    return disk_percent, disk_used_gb, disk_total_gb, disk_free_gb
