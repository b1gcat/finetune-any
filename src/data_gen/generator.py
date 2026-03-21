"""训练数据生成器 - 加载和处理训练数据"""
import json
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class QA:
    """问答对"""
    instruction: str
    input: str
    output: str

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output
        }

class DataGenerator:
    """数据生成器"""

    def __init__(self):
        self.qa_pairs: List[QA] = []

    def load_from_jsonl(self, jsonl_path: str) -> List[QA]:
        """从JSONL文件加载问答对"""
        qa_list = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    qa_list.append(QA(
                        instruction=data.get('instruction', ''),
                        input=data.get('input', ''),
                        output=data.get('output', '')
                    ))
        print(f"从 {jsonl_path} 加载了 {len(qa_list)} 个问答对")
        return qa_list

    def load_dataset_dir(self, dataset_dir: str, train_name: str = "train.jsonl", 
                         test_name: str = "test.jsonl", test_ratio: float = 0.0) -> tuple:
        """从数据集目录加载训练和测试数据
        
        Args:
            dataset_dir: 数据集目录路径
            train_name: 训练数据文件名
            test_name: 测试数据文件名
            test_ratio: 如果没有测试集，按此比例从训练集分割
            
        Returns:
            (train_qa_list, test_qa_list)
        """
        dataset_path = Path(dataset_dir)
        train_path = dataset_path / train_name
        test_path = dataset_path / test_name
        
        train_qa = self.load_from_jsonl(str(train_path))
        
        if test_path.exists():
            test_qa = self.load_from_jsonl(str(test_path))
        else:
            test_qa = []
            if test_ratio > 0 and train_qa:
                import random
                random.shuffle(train_qa)
                test_size = int(len(train_qa) * test_ratio)
                test_qa = train_qa[:test_size]
                train_qa = train_qa[test_size:]
                print(f"从训练集分割 {len(test_qa)} 个测试样本")
        
        return train_qa, test_qa

    def save_to_jsonl(self, qa_list: List[QA], output_path: str):
        """保存为JSONL格式"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in qa_list:
                f.write(json.dumps(qa.to_dict(), ensure_ascii=False) + '\n')

        print(f"已保存 {len(qa_list)} 个问答对到: {output_path}")
