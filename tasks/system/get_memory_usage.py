"""Read current memory usage from the host machine."""

import psutil

from core.task import task


@task(
    outputs=[
        "memory_percent",
        "memory_used_gb",
        "memory_total_gb",
        "memory_available_gb",
    ],
    display_name="Get Memory Usage",
    description="Read current RAM utilization and capacity from the host machine",
    category="system",
    output_types={
        "memory_percent": "float",
        "memory_used_gb": "float",
        "memory_total_gb": "float",
        "memory_available_gb": "float",
    },
    is_collapsed=True,
    parameters={},
)
def get_memory_usage():
    """
    Read current memory usage from the host machine.

    Returns:
        memory_percent: RAM usage percentage (0–100)
        memory_used_gb: Used RAM in gigabytes
        memory_total_gb: Total installed RAM in gigabytes
        memory_available_gb: Available RAM in gigabytes
    """
    vm = psutil.virtual_memory()

    _GB = 1024**3
    memory_percent = vm.percent
    memory_used_gb = round(vm.used / _GB, 2)
    memory_total_gb = round(vm.total / _GB, 2)
    memory_available_gb = round(vm.available / _GB, 2)

    return memory_percent, memory_used_gb, memory_total_gb, memory_available_gb
