import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.soft_grpo_manifest import load_manifest, record_stage1_completion  # noqa: E402
from pipeline.soft_grpo_dataset import SoftGrpoDatasetBuilder  # noqa: E402


def test_record_stage1_completion(tmp_path):
    manifest_path = tmp_path / "models" / "phase3_hybrid" / "NQ" / "phase3_stage_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    record_stage1_completion(
        path=manifest_path,
        market="NQ",
        model_checkpoint="models/phase3_hybrid/NQ/phase3_hybrid_final.zip",
        vecnorm_path="models/vecnormalize/phase3.pkl",
        total_timesteps=1000,
        llm_config={"name": "meta-llama/Meta-Llama-3-8B", "local_path": "/models/llama3"},
        llm_stats={"total_queries": 5},
        hybrid_stats={"total_decisions": 10},
        lora_path="models/phase3_hybrid/NQ/stage1_lora",
        experience_path="data/soft_grpo/NQ/stage1_experience.jsonl",
        experience_count=3,
        test_mode=False,
    )

    loaded = load_manifest(manifest_path)
    assert loaded["stage1"]["model_checkpoint"].endswith("phase3_hybrid_final.zip")
    assert loaded["stage1"]["experience_samples"] == 3
    assert loaded["stage1"]["lora_adapter_path"].endswith("stage1_lora")


def test_dataset_builder_creates_outputs(tmp_path):
    dataset_root = tmp_path / "data" / "soft_grpo"
    model_root = tmp_path / "models" / "phase3_hybrid"
    model_root.mkdir(parents=True, exist_ok=True)

    experience_file = dataset_root / "NQ" / "stage1_experience.jsonl"
    experience_file.parent.mkdir(parents=True, exist_ok=True)
    sample_exp = {
        "id": 0,
        "prompt": "Sample trading prompt",
        "response": "HOLD | 0.5 | sample",
        "action": 0,
        "observation": [0.0, 1.0],
        "position_state": {"position": 0},
        "market_context": {"market": "NQ"},
        "timestamp": 123456.0,
        "outcome": {"reward": 1.0, "pnl": 50.0},
    }
    with experience_file.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(sample_exp) + "\n")

    config = {
        "soft_grpo": {
            "model_root": str(model_root),
            "dataset_root": str(dataset_root),
            "experience_filename": "stage1_experience.jsonl",
            "train_filename": "train.parquet",
            "val_filename": "val.parquet",
            "jsonl_snapshot": "samples.jsonl",
            "metadata_filename": "meta.json",
        },
        "dataset": {
            "ability_label": "trading",
            "data_source": "phase3_stage1",
            "include_failures": True,
            "val_split": 0.5,
            "max_samples": 10,
            "action_names": {0: "HOLD"},
        },
        "runner": {"args": {}, "test_overrides": {}},
    }

    builder = SoftGrpoDatasetBuilder(config=config, project_root=tmp_path)
    manifest = {
        "stage1": {
            "experience_path": str(experience_file),
        }
    }
    info = builder.build("NQ", manifest=manifest, test_mode=True)

    assert Path(info.train_path).exists()
    assert Path(info.val_path).exists()
    assert Path(info.jsonl_path).exists()
