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

    model_name: str = "qwen3:0.6b"
    model_path: Optional[str] = None
    data_path: str = "./output/temp/train.jsonl"
    output_dir: str = "./output"
    max_length: int = 512
    batch_size: int = 1
    learning_rate: float = 2e-4
    num_epochs: int = 3
    accumulation_steps: int = 16
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    deepspeed: Optional[str] = None
    device: str = "cuda"

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
        "qwen3:0.6b": "Qwen/Qwen3-0.6B",
    }

    def __init__(self, config: FinetuneConfig):
        self.config = config

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
            "--device",
            self.config.device,
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
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
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
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    device_map = "auto" if device == "cuda" else "cpu"
    print(f"使用设备: {device}, device_map: {device_map}")

    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model_name_map = {
        "qwen3:0.6b": "Qwen/Qwen3-0.6B",
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )

    print(f"加载数据: {args.data_path}")
    raw_data = load_data(args.data_path)
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(
        lambda x: preprocess(x, tokenizer, args.max_length),
        remove_columns=dataset.column_names,
        batched=False
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
        bf16=device == "cuda",
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
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

    def evaluate(self, data_path: Optional[str] = None, use_base_model: bool = False) -> dict:
        """评估模型"""
        import json

        if data_path is None:
            data_path = self.config.data_path

        eval_script = Path(self.config.output_dir) / "temp" / "evaluate.py"
        eval_script.parent.mkdir(parents=True, exist_ok=True)

        if use_base_model:
            model_name_map = {
                "qwen3:0.6b": "Qwen/Qwen3-0.6B",
            }
            base_model_id = model_name_map.get(
                self.config.model_name, self.config.model_name
            )
            model_load = f'''
from modelscope import snapshot_download
print(f"从ModelScope下载基础模型: {base_model_id}")
model_path = snapshot_download("{base_model_id}")
print(f"模型已缓存: {{model_path}}")
'''
        else:
            adapter_path = (
                self.config.model_path
                if self.config.model_path
                else self.get_adapter_path()
            )
            model_load = f'''
model_path = "{adapter_path}"
'''

        script_content = f'''#!/usr/bin/env python3
import os
os.environ["HF_ENDPOINT"] = "https://modelscope.cn"

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

device = "{self.config.device}"
device_map = "auto" if device == "cuda" else "cpu"
print(f"[INFO] 使用设备: {{device}}, device_map: {{device_map}}")

if device == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"[INFO] CPU threads: {{torch.get_num_threads()}}")

data_path = "{data_path}"
output_path = data_path.replace(".jsonl", "_eval_result.jsonl")

{model_load}
torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16
print(f"[INFO] 加载模型: {{model_path}}")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True
)
model.eval()

print(f"[INFO] 加载测试数据: {{data_path}}")
with open(data_path, "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

print(f"[INFO] 开始评估 (共{{len(test_data)}}题)\\n")

prompts = []
for item in test_data:
    instruction = item.get("instruction", "根据以下文档内容回答问题。回答时说明来源文档。")
    input_text = item.get("input", "")
    prompt = f"{{instruction}}\\n\\n{{input_text}}"
    prompts.append(prompt)

inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
batch_size = 16

results = []
with torch.no_grad():
    for i in range(0, len(prompts), batch_size):
        batch_inputs = {{k: v[i:i+batch_size] for k, v in inputs.items()}}
        outputs = model.generate(**batch_inputs, max_new_tokens=150, do_sample=False)
        progress = min(i + batch_size, len(prompts))
        print(f"[进度] {{progress}}/{{len(prompts)}}", end="\\r")
        for j, output in enumerate(outputs):
            idx = i + j
            response = tokenizer.decode(output, skip_special_tokens=True).replace(prompts[idx], "").strip()
            results.append({{"response": response, "prompt": prompts[idx]}})

print("\\n[INFO] 评估回答中...")

def calc_similarity(text1, text2):
    text1 = set(text1.lower().split())
    text2 = set(text2.lower().split())
    if not text1 or not text2:
        return 0.0
    return len(text1 & text2) / len(text1 | text2)

eval_results = []
correct = 0
for item, result in zip(test_data, results):
    expected = item.get("output", "")
    response = result["response"]

    sim = calc_similarity(response, expected)
    is_correct = sim >= 0.5

    if is_correct:
        correct += 1

    eval_results.append({{
        "instruction": item.get("instruction", ""),
        "input": item.get("input", ""),
        "expected": expected,
        "response": response,
        "similarity": round(sim, 3),
        "correct": is_correct
    }})

accuracy = correct / len(test_data) * 100

print("\\n" + "=" * 60)
print(f"[汇总] 正确: {{correct}}/{{len(test_data)}} ({{accuracy:.1f}}%)")
print("=" * 60)

with open(output_path, "w", encoding="utf-8") as f:
    for r in eval_results:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")

print(f"[INFO] 评估结果已保存: {{output_path}}")

for i, r in enumerate(eval_results[:5]):
    status = "✓" if r["correct"] else "✗"
    print(f"\\n[{{status}}] {{i+1}}. {{r['input'][:40]}}...")
    print(f"  预期: {{r['expected'][:50]}}")
    print(f"  回答: {{r['response'][:50]}}")
    print(f"  相似度: {{r['similarity']}}")

if len(eval_results) > 5:
    print(f"\\n... 还有 {{len(eval_results)-5}} 条结果")

print("\\n[INFO] 评估完成!")
'''
        eval_script.write_text(script_content)
        process = subprocess.Popen(
            ["python", str(eval_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout:
            for line in process.stdout:
                print(line, end="")
        process.wait()
        if process.returncode != 0:
            print(f"评估失败")
        with open(data_path, "r", encoding="utf-8") as f:
            count = len([json.loads(line) for line in f])
        return {"test_count": count}
