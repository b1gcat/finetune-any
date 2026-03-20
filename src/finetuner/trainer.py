"""微调管理器 - 调用Xtuner执行微调"""

import os
import subprocess
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import shutil

os.environ["HF_ENDPOINT"] = "https://modelscope.cn"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


@dataclass
class FinetuneConfig:
    """微调配置"""

    model_name: str = "Qwen2.5-0.5B"
    model_path: str = None
    data_path: str = "./output/temp/train.jsonl"
    output_dir: str = "./output"
    max_length: int = 2048
    batch_size: int = 1
    learning_rate: float = 2e-4
    num_epochs: int = 3
    accumulation_steps: int = 16
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    deepspeed: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_name_or_path": self.model_name,
            "data_path": self.data_path,
            "output_dir": self.output_dir,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "accumulation_steps": self.accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
        }


class Finetuner:
    """微调管理器"""

    SUPPORTED_MODELS = {
        "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
        "Qwen2.5-1.8B": "Qwen/Qwen2.5-1.8B",
        "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    }

    def __init__(self, config: FinetuneConfig):
        self.config = config
        self._check_xtuner()

    def _check_xtuner(self):
        """检查Xtuner是否安装"""
        try:
            subprocess.run(["xtuner", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("警告: Xtuner未安装，将使用transformers直接微调")
            self.use_xtuner = False
        else:
            self.use_xtuner = True

    def prepare_data(self) -> str:
        """准备训练数据"""
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"训练数据不存在: {data_path}")

        work_dir = Path("output/temp/data")
        work_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(data_path, work_dir / "train.jsonl")
        print(f"数据已准备: {work_dir / 'train.jsonl'}")

        return str(work_dir / "train.jsonl")

    def train(self):
        """执行微调训练"""
        print(f"开始微调: {self.config.model_name}")
        print(
            f"训练参数: batch_size={self.config.batch_size}, lr={self.config.learning_rate}, epochs={self.config.num_epochs}"
        )

        if self.use_xtuner:
            self._train_with_xtuner()
        else:
            self._train_with_transformers()

    def _train_with_transformers(self):
        """使用transformers微调"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_script = output_dir / "temp" / "train.py"
        train_script.parent.mkdir(parents=True, exist_ok=True)
        self._generate_train_script(train_script)

        cmd = [
            "python",
            str(train_script),
            "--model_name",
            self.config.model_name,
            "--data_path",
            self.config.data_path,
            "--output_dir",
            self.config.output_dir,
            "--num_epochs",
            str(self.config.num_epochs),
            "--batch_size",
            str(self.config.batch_size),
            "--learning_rate",
            str(self.config.learning_rate),
            "--max_length",
            str(self.config.max_length),
        ]
        if self.config.model_path:
            cmd.extend(["--model_path", self.config.model_path])
        print(f"运行训练脚本: {train_script}")
        subprocess.run(cmd, check=True)

    def _generate_train_script(self, output_path: Path):
        """生成训练脚本"""
        script = '''#!/usr/bin/env python3
"""Qwen微调训练脚本"""
import os
os.environ["HF_ENDPOINT"] = "https://modelscope.cn"

import argparse
import json
from pathlib import Path
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch

def load_data(data_path):
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def preprocess(example, tokenizer, max_length):
    prompt = f"{example['instruction']}\\n\\n{example['input']}"
    text = f"{prompt}\\n\\n{example['output']}"
    result = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None, help="本地模型路径(优先于model_name)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    model_name_map = {
        "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
        "Qwen2.5-1.8B": "Qwen/Qwen2.5-1.8B",
        "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    }
    model_id = model_name_map.get(args.model_name, args.model_name)
    
    if args.model_path and Path(args.model_path).exists():
        model_path = args.model_path
        print(f"使用本地模型: {model_path}")
    else:
        print(f"从ModelScope下载模型: {model_id}")
        model_path = snapshot_download(model_id)
        print(f"模型已缓存到: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        dtype=torch.float32,
        trust_remote_code=True
    )

    print(f"加载数据: {args.data_path}")
    raw_data = load_data(args.data_path)
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(
        lambda x: preprocess(x, tokenizer, args.max_length),
        remove_columns=dataset.column_names
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        report_to="none",
        fp16=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("开始训练...")
    trainer.train()
    print("训练完成，保存模型...")

    adapter_path = f"{args.output_dir}/adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA适配器已保存: {adapter_path}")

if __name__ == "__main__":
    main()
'''
        output_path.write_text(script)

    def get_adapter_path(self) -> str:
        """获取适配器路径"""
        return str(Path(self.config.output_dir) / "adapter")
