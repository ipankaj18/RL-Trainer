#!/usr/bin/env python3
"""Launch the hardware-maximized hybrid LLM/GPU validation run."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Local imports after sys.path update
from testing_framework import TestConfig, TestingFramework, create_test_config  # type: ignore  # noqa: E402
from model_utils import detect_available_markets  # type: ignore  # noqa: E402


PRESETS: Dict[str, Dict[str, float | int]] = {
    "fast": {
        "timesteps_reduction": 0.05,  # 5% of production timesteps
        "vectorized_envs": 12,
        "batch_size": 512,
    },
    "heavy": {
        "timesteps_reduction": 0.15,  # 15% of production timesteps
        "vectorized_envs": 24,
        "batch_size": 1024,
    },
}


def _infer_market(requested: Optional[str]) -> str:
    """Return a market symbol from CLI input or detected data files."""
    if requested:
        return requested.upper()

    detected = detect_available_markets(str(PROJECT_ROOT / "data"))
    if not detected:
        raise ValueError("No market data found under data/. Run data processing first.")

    if len(detected) == 1:
        return detected[0]["market"].upper()

    symbols = ", ".join(item["market"] for item in detected)
    raise ValueError(f"Multiple markets detected ({symbols}); pass --market to select one.")


def _build_config(market: str, preset: str, overrides: Dict[str, float | int]) -> TestConfig:
    """Create the TestingFramework configuration for the requested preset."""
    preset_values = PRESETS[preset].copy()
    preset_values.update(overrides)

    return create_test_config(
        mode="hardware_maximized",
        market=market,
        vectorized_envs=int(preset_values["vectorized_envs"]),
        batch_size=int(preset_values["batch_size"]),
        timesteps_reduction=float(preset_values["timesteps_reduction"]),
        device="auto",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run hybrid LLM/GPU hardware-maximized validation.")
    parser.add_argument("--market", help="Market symbol to test (e.g., NQ, ES).")
    parser.add_argument(
        "--preset",
        choices=PRESETS.keys(),
        default="fast",
        help="Preset controlling timesteps, batch size, and env count.",
    )
    parser.add_argument("--timesteps-reduction", type=float, dest="timesteps_reduction", help="Override timesteps reduction fraction (0-1).")
    parser.add_argument("--batch-size", type=int, dest="batch_size", help="Override PPO batch size.")
    parser.add_argument("--vectorized-envs", type=int, dest="vectorized_envs", help="Override number of parallel envs.")
    return parser.parse_args()


def main() -> None:
    """Entrypoint invoked by the main menu."""
    args = parse_args()
    project_llm_dir = PROJECT_ROOT / "Phi-3-mini-4k-instruct"

    # Surface LLM + CUDA expectations up front
    if not project_llm_dir.exists():
        print(
            "[WARN] Phi-3-mini-4k-instruct folder not found at project root. "
            "Hybrid validation expects the local model; download it before running."
        )

    cuda_root = os.environ.get("CUDA_ROOT")
    if cuda_root:
        print(f"[INFO] Using CUDA_ROOT={cuda_root}")
    else:
        print("[WARN] CUDA_ROOT not set; ensure your venv activation exports it for GPU tests.")

    market = _infer_market(args.market)

    overrides: Dict[str, float | int] = {}
    if args.timesteps_reduction is not None:
        overrides["timesteps_reduction"] = args.timesteps_reduction
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.vectorized_envs is not None:
        overrides["vectorized_envs"] = args.vectorized_envs

    config = _build_config(market=market, preset=args.preset, overrides=overrides)

    print(
        f"[INFO] Launching hybrid LLM/GPU validation on {market} "
        f"preset={args.preset} (vec_envs={config.vectorized_envs}, batch={config.batch_size}, "
        f"timesteps_reduction={config.timesteps_reduction})"
    )

    # Ensure relative paths (llm_config, logs) resolve from the project root
    os.chdir(PROJECT_ROOT)

    framework = TestingFramework(config)
    framework.run_hardware_maximized_validation()


if __name__ == "__main__":
    main()
