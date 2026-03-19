"""数据生成器测试"""
import pytest
from src.data_gen.generator import DataGenerator, QA

def test_qa_to_dict():
    qa = QA(
        instruction="测试指令",
        input="测试问题",
        output="测试回答"
    )
    d = qa.to_dict()
    assert d["instruction"] == "测试指令"
    assert d["input"] == "测试问题"
    assert d["output"] == "测试回答"

def test_generate_from_text():
    gen = DataGenerator()
    text = "第1条 为了规范XXX行为，制定本标准。"
    qa_list = gen.generate_from_text(text, "《测试文档》")
    assert len(qa_list) > 0
