"""微调管理器测试"""
import pytest
from src.finetuner.trainer import FinetuneConfig, Finetuner

def test_finetune_config_to_dict():
    config = FinetuneConfig(
        model_name="Qwen2.5-0.5B",
        batch_size=2,
        num_epochs=5
    )
    d = config.to_dict()
    assert d["model_name_or_path"] == "Qwen2.5-0.5B"
    assert d["batch_size"] == 2
    assert d["num_epochs"] == 5

def test_supported_models():
    config = FinetuneConfig()
    assert "Qwen2.5-0.5B" in Finetuner.SUPPORTED_MODELS
    assert "Qwen2.5-1.8B" in Finetuner.SUPPORTED_MODELS
