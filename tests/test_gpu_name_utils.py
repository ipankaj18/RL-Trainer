import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "jax_migration" / "gpu_name_utils.py"

spec = importlib.util.spec_from_file_location("gpu_name_utils", MODULE_PATH)
gpu_name_utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None  # Narrow mypy/pyright concerns
spec.loader.exec_module(gpu_name_utils)

sanitize_gpu_name = gpu_name_utils.sanitize_gpu_name
get_gpu_name = gpu_name_utils.get_gpu_name


def test_sanitize_common_models():
    assert sanitize_gpu_name("NVIDIA RTX 4000 Ada Generation") == "RTX4000AdaGeneration"
    assert sanitize_gpu_name("NVIDIA GeForce RTX 5090") == "RTX5090"
    assert sanitize_gpu_name("NVIDIA GeForce RTX 4090") == "RTX4090"
    assert sanitize_gpu_name("NVIDIA GeForce RTX 3060 Ti") == "RTX3060Ti"
    assert sanitize_gpu_name("NVIDIA Tesla V100") == "V100"


def test_sanitize_handles_lowercase_and_gaps():
    assert sanitize_gpu_name("nvidia rtx a6000") == "RTXA6000"
    assert sanitize_gpu_name("   RTX 6000 Ada Generation  ") == "RTX6000AdaGeneration"


def test_override_short_circuits_detection():
    override_value = "Custom GPU 123"
    assert get_gpu_name(override_name=override_value) == "CustomGPU123"
