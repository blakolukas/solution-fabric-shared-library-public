"""Read current GPU usage from the host machine."""

import json
import subprocess

from core.task import task


@task(
    outputs=["gpus_json"],
    display_name="Get GPU Usage",
    description="Read current GPU utilization, memory, and temperature from the host machine. Supports NVIDIA (nvidia-smi) and AMD (rocm-smi). Returns a JSON array of per-GPU stats; empty array when no GPU is present.",
    category="system",
    output_types={"gpus_json": "str"},
    is_collapsed=True,
    parameters={},
)
def get_gpu_usage():
    """
    Read current GPU usage from the host machine.

    Queries NVIDIA GPUs via nvidia-smi first, then falls back to AMD GPUs
    via rocm-smi. Returns one entry per detected GPU so the caller can
    display or aggregate them independently.

    Returns:
        gpus_json: JSON array of GPU stat objects. Each object contains:
            name, index, memory_total_mb, memory_used_mb, memory_percent,
            utilization_percent, temperature_c.
            Empty array [] when no GPU hardware is detected.
    """
    raw_gpus = _detect_gpus()

    result = []
    for g in raw_gpus:
        memory_total = g["memory_total_mb"] or 0.0
        memory_used = g["memory_used_mb"] or 0.0
        memory_percent = round(
            (memory_used / memory_total * 100) if memory_total > 0 else 0.0, 1
        )
        result.append(
            {
                "name": g["name"],
                "index": g["index"],
                "memory_total_mb": round(memory_total, 1),
                "memory_used_mb": round(memory_used, 1),
                "memory_percent": memory_percent,
                "utilization_percent": (
                    round(g["utilization_percent"], 1)
                    if g["utilization_percent"] is not None
                    else 0.0
                ),
                "temperature_c": (
                    round(g["temperature_c"], 1)
                    if g["temperature_c"] is not None
                    else 0.0
                ),
            }
        )

    return json.dumps(result)


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
