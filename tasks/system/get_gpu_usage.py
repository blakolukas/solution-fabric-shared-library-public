"""Read current GPU usage from the host machine."""

import subprocess

from core.task import task


@task(
    outputs=[
        "gpu_name",
        "gpu_count",
        "gpu_memory_total_mb",
        "gpu_memory_used_mb",
        "gpu_memory_percent",
        "gpu_utilization_percent",
        "gpu_temperature_c",
    ],
    display_name="Get GPU Usage",
    description="Read current GPU utilization, memory, and temperature from the host machine. Supports NVIDIA (nvidia-smi) and AMD (rocm-smi). Returns zero/null values when no GPU is present.",
    category="system",
    output_types={
        "gpu_name": "str",
        "gpu_count": "int",
        "gpu_memory_total_mb": "float",
        "gpu_memory_used_mb": "float",
        "gpu_memory_percent": "float",
        "gpu_utilization_percent": "float",
        "gpu_temperature_c": "float",
    },
    is_collapsed=True,
    parameters={},
)
def get_gpu_usage():
    """
    Read current GPU usage from the host machine.

    Queries NVIDIA GPUs via nvidia-smi first, then falls back to AMD GPUs
    via rocm-smi. Aggregates multi-GPU systems: name is the primary GPU,
    memory/utilization/temperature are averaged across all detected GPUs.
    Returns safe zero/null defaults when no GPU hardware is detected.

    Returns:
        gpu_name: Primary GPU name, or "No GPU" if none detected
        gpu_count: Number of GPUs detected
        gpu_memory_total_mb: Total GPU memory in MB (0.0 if no GPU)
        gpu_memory_used_mb: Used GPU memory in MB (0.0 if no GPU)
        gpu_memory_percent: GPU memory usage percentage (0.0 if no GPU)
        gpu_utilization_percent: GPU utilization percentage (0.0 if no GPU)
        gpu_temperature_c: GPU temperature in Celsius (0.0 if no GPU)
    """
    gpus = _detect_gpus()

    if not gpus:
        return "No GPU", 0, 0.0, 0.0, 0.0, 0.0, 0.0

    gpu_name = gpus[0]["name"]
    gpu_count = len(gpus)
    gpu_memory_total_mb = sum(g["memory_total_mb"] for g in gpus)
    gpu_memory_used_mb = sum(g["memory_used_mb"] for g in gpus)
    gpu_memory_percent = round(
        (
            (gpu_memory_used_mb / gpu_memory_total_mb * 100)
            if gpu_memory_total_mb > 0
            else 0.0
        ),
        1,
    )
    utilization_values = [
        g["utilization_percent"] for g in gpus if g["utilization_percent"] is not None
    ]
    gpu_utilization_percent = (
        round(sum(utilization_values) / len(utilization_values), 1)
        if utilization_values
        else 0.0
    )
    temperature_values = [
        g["temperature_c"] for g in gpus if g["temperature_c"] is not None
    ]
    gpu_temperature_c = round(max(temperature_values), 1) if temperature_values else 0.0

    return (
        gpu_name,
        gpu_count,
        round(gpu_memory_total_mb, 1),
        round(gpu_memory_used_mb, 1),
        gpu_memory_percent,
        gpu_utilization_percent,
        gpu_temperature_c,
    )


def _safe_float(value: str, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _detect_gpus() -> list:
    """Try NVIDIA first, then AMD. Returns list of GPU dicts."""
    gpus = _detect_nvidia()
    if gpus:
        return gpus
    return _detect_amd()


def _detect_nvidia() -> list:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,index,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            memory_total = _safe_float(parts[2])
            memory_used = _safe_float(parts[3])
            gpus.append(
                {
                    "name": parts[0],
                    "index": int(parts[1]) if parts[1].isdigit() else 0,
                    "memory_total_mb": memory_total,
                    "memory_used_mb": memory_used,
                    "utilization_percent": (
                        _safe_float(parts[4]) if len(parts) > 4 else None
                    ),
                    "temperature_c": _safe_float(parts[5]) if len(parts) > 5 else None,
                }
            )
        return gpus
    except Exception:
        return []


def _detect_amd() -> list:
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        import json

        data = json.loads(result.stdout)
        gpus = []
        for key, info in data.items():
            if not key.startswith("card"):
                continue
            memory_total = _safe_float(info.get("VRAM Total Memory (B)", 0)) / (
                1024 * 1024
            )
            memory_used = _safe_float(info.get("VRAM Total Used Memory (B)", 0)) / (
                1024 * 1024
            )
            gpus.append(
                {
                    "name": info.get("Card Series", f"AMD GPU {key}"),
                    "index": int(key.replace("card", "")) if key[4:].isdigit() else 0,
                    "memory_total_mb": memory_total,
                    "memory_used_mb": memory_used,
                    "utilization_percent": _safe_float(info.get("GPU use (%)", None)),
                    "temperature_c": None,
                }
            )
        return gpus
    except Exception:
        return []
