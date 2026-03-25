"""模型转换器 - 将微调模型转换为Ollama格式"""
import os
import subprocess
from pathlib import Path
from typing import Optional

class ModelConverter:
    """模型转换器"""

    def __init__(self, base_model: str, adapter_path: str, output_dir: str = "./output", device: str = "cuda"):
        self.base_model = base_model
        self.adapter_path = Path(adapter_path)
        self.output_dir = Path(output_dir)
        self.merged_dir = self.output_dir / "merged"
        self.device = device

    def merge_adapter(self) -> str:
        """合并LoRA适配器到基础模型"""
        print("合并LoRA适配器到基础模型...")

        self.merged_dir.mkdir(parents=True, exist_ok=True)

        device = "{self.device}"
        device_map = "auto" if device == "cuda" else "cpu"
        torch_dtype = "torch.bfloat16" if device == "cuda" else "torch.float32"

        merge_script = self.output_dir / "merge.py"
        merge_script.write_text(f'''#!/usr/bin/env python3
import os
os.environ["HF_ENDPOINT"] = "https://modelscope.cn"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

device = "{device}"
device_map = "auto" if device == "cuda" else "cpu"
torch_dtype = {torch_dtype}
print(f"使用设备: {{device}}, device_map: {{device_map}}")

adapter_path = "{self.adapter_path}"
output_path = "{self.merged_dir}"

adapter_config = Path(adapter_path) / "adapter_config.json"
if adapter_config.exists():
    print("检测到LoRA适配器，合并模型...")
    from modelscope import snapshot_download
    from peft import PeftModel
    
    model_name_map = {{
        "qwen3:0.6b": "Qwen/Qwen3-0.6B",
    }}
    base_model_id = model_name_map.get("{self.base_model}", "{self.base_model}")
    print(f"从ModelScope下载基础模型: {{base_model_id}}")
    model_path = snapshot_download(base_model_id)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True
    )
    print(f"加载LoRA适配器: {{adapter_path}}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.merge_and_unload()
    print("适配器合并完成")
else:
    print("检测到完整模型，直接使用...")
    model_path = Path(adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True
    )

print(f"保存模型到: {{output_path}}")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

import subprocess
size = subprocess.check_output(["du", "-sh", output_path]).decode().split()[0]
print(f"模型大小: {{size}}")
print("完成!")
'''

        return self.create_ollama_model(model_name)

    def create_ollama_model(self, model_name: str) -> str:
        """创建 Ollama 模型"""
        ollama_dir = self.merged_dir

        modelfile_content = f'''FROM ./merged
PARAMETER temperature 0.1
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER repeat_penalty 1.1
'''

        modelfile_path = ollama_dir / "Modelfile"
        modelfile_path.write_text(modelfile_content)

        print(f"创建Ollama模型: {model_name}")
        print(f"Modelfile: {modelfile_path}")
        print(f"模型目录: {ollama_dir}")

        try:
            subprocess.run([
                "ollama", "create", model_name,
                "-f", str(modelfile_path),
            ], check=True, cwd=str(ollama_dir))
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("提示: Ollama未安装或不在PATH中，请手动运行:")
            print(f"  cd {ollama_dir}")
            print(f"  ollama create {model_name} -f Modelfile")

        return str(ollama_dir)

    def full_pipeline(self, model_name: str) -> str:
        """完整转换流程"""
        self.merge_adapter()
        return self.create_ollama_model(model_name)