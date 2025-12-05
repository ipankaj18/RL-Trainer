"""
GPU name detection and sanitization helpers.

This module centralizes GPU name handling so the stress test and training
scripts can produce consistent, descriptive hardware profile filenames.
"""

from __future__ import annotations

import os
import subprocess
from collections import Counter
from typing import List, Optional, Tuple


def sanitize_gpu_name(raw_name: str) -> str:
    """
    Normalize a raw GPU name into a filename-safe prefix.

    Examples:
        "NVIDIA RTX 4000 Ada Generation" -> "RTX4000AdaGeneration"
        "NVIDIA GeForce RTX 5090" -> "RTX5090"
        "NVIDIA GeForce RTX 3060 Ti" -> "RTX3060Ti"
    """
    if not raw_name:
        return "UNKNOWN"

    lower_name = raw_name.lower().strip()
    known_mappings = {
        "nvidia rtx 4000 ada generation": "RTX4000AdaGeneration",
        "rtx 4000 ada generation": "RTX4000AdaGeneration",
        "nvidia rtx a4000": "RTXA4000",
        "nvidia rtx a6000": "RTXA6000",
        "nvidia rtx 6000 ada generation": "RTX6000AdaGeneration",
        "nvidia geforce rtx 5090": "RTX5090",
        "nvidia geforce rtx 4090": "RTX4090",
    }

    if lower_name in known_mappings:
        return known_mappings[lower_name]

    sanitized = raw_name
    prefixes_to_remove = ["NVIDIA", "GeForce", "Quadro", "Tesla"]
    for prefix in prefixes_to_remove:
        sanitized = sanitized.replace(prefix, "")
        sanitized = sanitized.replace(prefix.lower(), "")
        sanitized = sanitized.replace(prefix.upper(), "")

    sanitized = sanitized.replace(" ", "").replace("-", "").replace("_", "")
    sanitized = "".join(c for c in sanitized if c.isalnum())
    sanitized = sanitized.strip()

    return sanitized if sanitized else "UNKNOWN"


def get_gpu_name(device_id: int = 0, debug: bool = False, override_name: Optional[str] = None) -> str:
    """
    Detect and sanitize GPU name from multiple sources.

    Order of precedence:
    1. Explicit override (argument or GPU_NAME_OVERRIDE env var)
    2. NVML
    3. JAX device info
    4. nvidia-smi
    """
    chosen_override = override_name or os.environ.get("GPU_NAME_OVERRIDE")
    if chosen_override:
        sanitized_override = sanitize_gpu_name(chosen_override)
        if debug:
            print(f"[DEBUG] GPU name override provided: '{chosen_override}' -> '{sanitized_override}'")
        return sanitized_override

    candidates: List[Tuple[str, str]] = []

    # NVML
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        device_name = pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()

        if isinstance(device_name, bytes):
            device_name = device_name.decode("utf-8")

        candidates.append(("nvml", device_name))
    except Exception as exc:  # pragma: no cover - environment dependent
        if debug:
            print(f"[DEBUG] NVML detection failed: {exc}")

    # JAX device info
    try:
        import jax

        devices = jax.devices()
        if devices and len(devices) > device_id:
            candidates.append(("jax", devices[device_id].device_kind))
    except Exception as exc:  # pragma: no cover - optional dependency
        if debug:
            print(f"[DEBUG] JAX detection failed: {exc}")

    # nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
                "-i",
                str(device_id),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        smi_name = (result.stdout or "").strip().splitlines()[0]
        if smi_name:
            candidates.append(("nvidia-smi", smi_name))
    except Exception as exc:  # pragma: no cover - external binary
        if debug:
            print(f"[DEBUG] nvidia-smi detection failed: {exc}")

    sanitized_candidates = [
        (source, raw, sanitize_gpu_name(raw)) for source, raw in candidates if raw
    ]

    if debug and sanitized_candidates:
        print("[DEBUG] GPU name candidates:")
        for source, raw, sanitized in sanitized_candidates:
            print(f"  {source}: '{raw}' -> '{sanitized}'")

    # Majority vote on sanitized names; fall back to first candidate
    valid_names = [san for _, _, san in sanitized_candidates if san != "UNKNOWN"]
    if valid_names:
        counts = Counter(valid_names)
        chosen = counts.most_common(1)[0][0]
        if debug and len(counts) > 1:
            print(f"[DEBUG] Candidate disagreement resolved to '{chosen}' via majority vote")
        return chosen

    if sanitized_candidates:
        return sanitized_candidates[0][2]

    if debug:
        print("[DEBUG] No GPU names detected; returning 'UNKNOWN'")
    return "UNKNOWN"
