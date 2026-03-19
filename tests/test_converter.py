"""模型转换器测试"""
import pytest
from pathlib import Path
from src.converter.converter import ModelConverter

def test_converter_init():
    converter = ModelConverter(
        base_model="Qwen2.5-0.5B",
        adapter_path="./output/adapter",
        output_dir="./output"
    )
    assert converter.base_model == "Qwen2.5-0.5B"
    assert converter.merged_dir.name == "merged"