# PDF微调程序实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现一个命令行微调程序，读取PDF目录→生成训练数据→CPU微调→转Ollama格式

**Architecture:** 采用模块化设计，分为PDF解析、数据生成、微调管理、模型转换四个核心模块，通过CLI统一入口，支持单步执行和全流程一键执行。

**Tech Stack:** Python 3.10+, pdfplumber, Xtuner, transformers, llama-cpp-python, Click

---

## 任务 1: 项目初始化

**Files:**
- Create: `main.py`
- Create: `config.yaml`
- Create: `requirements.txt`
- Create: `src/__init__.py`

**Step 1: 创建项目目录结构**

```bash
mkdir -p src/pdf_parser src/data_gen src/finetuner src/converter data/raw data/processed output/adapter output/merged output/ollama docs/plans
```

**Step 2: 创建 requirements.txt**

```python
pdfplumber>=0.10.0
python-docx>=0.8.0
markdown>=3.4.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
transformers>=4.35.0
torch>=2.0.0
accelerate>=0.20.0
bitsandbytes>=0.37.0
safetensors>=0.3.0
llama-cpp-python>=0.2.0
click>=8.0
pyyaml>=6.0
tqdm>=4.65
```

**Step 3: 创建 config.yaml**

```yaml
model:
  name: Qwen2.5-0.5B
  max_length: 2048

training:
  batch_size: 1
  learning_rate: 2e-4
  num_epochs: 3
  accumulation_steps: 16
  warmup_steps: 100

paths:
  data_dir: ./data
  output_dir: ./output
  cache_dir: ~/.cache/huggingface
```

**Step 4: 创建 src/__init__.py**

```python
"""PDF Finetune Toolkit - 从PDF自动生成训练数据并微调模型"""
__version__ = "0.1.0"
```

---

## 任务 2: 文档解析器模块

**Files:**
- Create: `src/doc_parser/__init__.py`
- Create: `src/doc_parser/parser.py`
- Create: `tests/test_doc_parser.py`

**Step 1: 创建 src/doc_parser/__init__.py**

```python
"""文档解析模块 - 支持 PDF/DOCX/TXT/Markdown"""
from .parser import DocumentParser, Document, Page

__all__ = ["DocumentParser", "Document", "Page"]
```

**Step 2: 创建 src/doc_parser/parser.py**

```python
"""文档解析器 - 读取目录下的所有文档并提取文本"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import docx
except ImportError:
    docx = None

try:
    import markdown
except ImportError:
    markdown = None

@dataclass
class Page:
    """文档页面/章节"""
    page_num: int
    text: str
    tables: List[List[List[str]]] = field(default_factory=list)

@dataclass
class Document:
    """文档"""
    filename: str
    filepath: str
    pages: List[Page] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def full_text(self) -> str:
        """获取完整文本"""
        return "\n\n".join(p.text for p in self.pages)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "metadata": self.metadata,
            "pages": [
                {"page_num": p.page_num, "text": p.text, "tables": p.tables}
                for p in self.pages
            ]
        }

class DocumentParser:
    """文档解析器 - 支持 PDF/DOCX/TXT/Markdown"""

    SUPPORTED_FORMATS = {".pdf", ".docx", ".txt", ".md", ".markdown"}

    def __init__(self, doc_dir: str):
        self.doc_dir = Path(doc_dir)
        if not self.doc_dir.exists():
            raise FileNotFoundError(f"文档目录不存在: {doc_dir}")

    def parse_single(self, doc_path: str) -> Document:
        """解析单个文档"""
        path = Path(doc_path)
        if not path.exists():
            raise FileNotFoundError(f"文档文件不存在: {doc_path}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._parse_pdf(path)
        elif suffix == ".docx":
            return self._parse_docx(path)
        elif suffix == ".txt":
            return self._parse_txt(path)
        elif suffix in {".md", ".markdown"}:
            return self._parse_markdown(path)
        else:
            raise ValueError(f"不支持的格式: {suffix}")

    def _parse_pdf(self, path: Path) -> Document:
        """解析PDF"""
        if pdfplumber is None:
            raise ImportError("请安装 pdfplumber: pip install pdfplumber")

        doc = Document(filename=path.name, filepath=str(path))

        with pdfplumber.open(path) as pdf:
            doc.metadata = {
                "title": pdf.metadata.get("Title", ""),
                "author": pdf.metadata.get("Author", ""),
                "page_count": len(pdf.pages)
            }

            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                doc.pages.append(Page(page_num=i, text=text, tables=tables))

        return doc

    def _parse_docx(self, path: Path) -> Document:
        """解析DOCX"""
        if docx is None:
            raise ImportError("请安装 python-docx: pip install python-docx")

        doc = Document(filename=path.name, filepath=str(path))

        doc_obj = docx.Document(path)
        doc.metadata = {
            "title": doc_obj.core_properties.title or "",
            "author": doc_obj.core_properties.author or "",
        }

        text_parts = []
        tables = []
        for para in doc_obj.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        for i, table in enumerate(doc_obj.tables, 1):
            table_data = [[cell.text for cell in row.cells] for row in table.rows]
            tables.append(table_data)
            text_parts.append(f"[表格 {i}]")

        doc.pages.append(Page(
            page_num=1,
            text="\n".join(text_parts),
            tables=tables
        ))

        return doc

    def _parse_txt(self, path: Path) -> Document:
        """解析TXT"""
        doc = Document(filename=path.name, filepath=str(path))
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        doc.pages.append(Page(page_num=1, text=text))
        return doc

    def _parse_markdown(self, path: Path) -> Document:
        """解析Markdown"""
        doc = Document(filename=path.name, filepath=str(path))

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        if markdown is not None:
            html = markdown.markdown(content)
            doc.metadata["html"] = html

        doc.pages.append(Page(page_num=1, text=content))
        return doc

    def parse_all(self, recursive: bool = False) -> List[Document]:
        """解析目录下所有支持的文档"""
        documents = []

        for suffix in self.SUPPORTED_FORMATS:
            pattern = f"**/*{suffix}" if recursive else f"*{suffix}"
            for doc_file in self.doc_dir.glob(pattern):
                try:
                    doc = self.parse_single(doc_file)
                    documents.append(doc)
                    print(f"✓ 解析完成: {doc.filename} ({len(doc.pages)} 页/节)")
                except Exception as e:
                    print(f"✗ 解析失败: {doc_file.name} - {e}")

        return documents

    def save_to_json(self, documents: List[Document], output_path: str):
        """保存解析结果到JSON"""
        data = [doc.to_dict() for doc in documents]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已保存到: {output_path}")
```

**Step 3: 创建 tests/test_doc_parser.py**

```python
"""文档解析器测试"""
import pytest
import tempfile
from pathlib import Path
from src.doc_parser.parser import DocumentParser, Document, Page

def test_document_full_text():
    doc = Document(
        filename="test.pdf",
        filepath="/tmp/test.pdf",
        pages=[Page(1, "第一页"), Page(2, "第二页")]
    )
    assert doc.full_text() == "第一页\n\n第二页"

def test_document_to_dict():
    doc = Document(
        filename="test.pdf",
        filepath="/tmp/test.pdf",
        pages=[Page(1, "测试内容")],
        metadata={"title": "测试文档"}
    )
    data = doc.to_dict()
    assert data["filename"] == "test.pdf"
    assert data["metadata"]["title"] == "测试文档"

def test_supported_formats():
    parser = DocumentParser("/tmp")
    assert ".pdf" in parser.SUPPORTED_FORMATS
    assert ".docx" in parser.SUPPORTED_FORMATS
    assert ".txt" in parser.SUPPORTED_FORMATS
    assert ".md" in parser.SUPPORTED_FORMATS
```

---

## 任务 3: 训练数据生成器模块

**Files:**
- Create: `src/data_gen/__init__.py`
- Create: `src/data_gen/generator.py`
- Create: `tests/test_data_gen.py`

**Step 1: 创建 src/data_gen/__init__.py**

```python
"""训练数据生成模块"""
from .generator import DataGenerator, QA

__all__ = ["DataGenerator", "QA"]
```

**Step 2: 创建 src/data_gen/generator.py**

```python
"""训练数据生成器 - 从标准/规定文档生成问答对"""
import json
import re
from dataclasses import dataclass
from typing import List, Dict
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

    def generate_from_text(self, text: str, context: str = "") -> List[QA]:
        """从文本生成问答对"""
        qa_list = []

        sections = self._split_sections(text)
        for section in sections:
            qa_list.extend(self._extract_from_section(section, context))

        return qa_list

    def _split_sections(self, text: str) -> List[str]:
        """分割文本为章节"""
        patterns = [
            r'第[一二三四五六七八九十百\d]+[章节条款]',
            r'\d+\.\d+',
            r'第[一二三四五六七八九十百\d]+条',
            r'\([a-zA-Z]\)',
        ]

        pattern = '|'.join(patterns)
        parts = re.split(f'({pattern})', text)

        sections = []
        for i in range(1, len(parts), 2):
            header = parts[i]
            content_start = i + 1
            content = ''.join(parts[content_start:content_start+1] if content_start < len(parts) else [])
            sections.append(f"{header} {content}".strip())

        return [s for s in sections if len(s) > 20]

    def _extract_from_section(self, section: str, context: str) -> List[QA]:
        """从章节提取问答对"""
        qa_list = []

        if len(section) < 30:
            return qa_list

        keywords = {
            r'定义|概念|是指': ('definition', '什么是'),
            r'要求|应当|必须|需要': ('requirement', '有什么要求'),
            r'禁止|不得|不准': ('prohibition', '有什么禁止'),
            r'责任|义务|职责': ('duty', '有什么责任'),
            r'适用于|适用范围': ('scope', '适用范围'),
            r'程序|流程|步骤': ('procedure', '程序是什么'),
        }

        for pattern, (qtype, question_prefix) in keywords.items():
            if re.search(pattern, section):
                instruction = f"根据以下文档内容回答问题。"
                question = f"{context}关于{qtype}，{section[:100]}... 具体内容是什么？"

                qa_list.append(QA(
                    instruction=instruction,
                    input=question,
                    output=section
                ))

        if not qa_list:
            qa_list.append(QA(
                instruction="根据以下文档内容回答问题。",
                input=f"{context}这段内容讲了什么？",
                output=section[:500]
            ))

        return qa_list

    def generate_from_documents(self, documents: List[Dict]) -> List[QA]:
        """从文档列表生成问答对"""
        all_qa = []

        for doc in documents:
            filename = doc.get('filename', '')
            context = f"《{filename}》"

            full_text = "\n".join(p.get('text', '') for p in doc.get('pages', []))
            qa_list = self.generate_from_text(full_text, context)
            all_qa.extend(qa_list)

        return all_qa

    def save_to_jsonl(self, qa_list: List[QA], output_path: str):
        """保存为JSONL格式"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in qa_list:
                f.write(json.dumps(qa.to_dict(), ensure_ascii=False) + '\n')

        print(f"已生成 {len(qa_list)} 个问答对，保存到: {output_path}")
```

**Step 3: 创建 tests/test_data_gen.py**

```python
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
```

---

## 任务 4: 微调管理器模块

**Files:**
- Create: `src/finetuner/__init__.py`
- Create: `src/finetuner/trainer.py`
- Create: `tests/test_finetuner.py`

**Step 1: 创建 src/finetuner/__init__.py**

```python
"""微调管理模块"""
from .trainer import Finetuner, FinetuneConfig

__all__ = ["Finetuner", "FinetuneConfig"]
```

**Step 2: 创建 src/finetuner/trainer.py**

```python
"""微调管理器 - 调用Xtuner执行微调"""
import os
import subprocess
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import shutil

@dataclass
class FinetuneConfig:
    """微调配置"""
    model_name: str = "Qwen2.5-0.5B"
    data_path: str = "./data/train.jsonl"
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

        work_dir = Path(self.config.output_dir) / "data"
        work_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(data_path, work_dir / "train.jsonl")
        print(f"数据已准备: {work_dir / 'train.jsonl'}")

        return str(work_dir / "train.jsonl")

    def train(self):
        """执行微调训练"""
        print(f"开始微调: {self.config.model_name}")
        print(f"训练参数: batch_size={self.config.batch_size}, lr={self.config.learning_rate}, epochs={self.config.num_epochs}")

        if self.use_xtuner:
            self._train_with_xtuner()
        else:
            self._train_with_transformers()

    def _train_with_transformers(self):
        """使用transformers微调"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_script = output_dir / "train.py"
        self._generate_train_script(train_script)

        print(f"运行训练脚本: {train_script}")
        subprocess.run([
            "python", str(train_script),
            "--model_name", self.config.model_name,
            "--data_path", self.config.data_path,
            "--output_dir", self.config.output_dir,
            "--num_epochs", str(self.config.num_epochs),
            "--batch_size", str(self.config.batch_size),
            "--learning_rate", str(self.config.learning_rate),
            "--max_length", str(self.config.max_length),
        ], check=True)

    def _generate_train_script(self, output_path: Path):
        """生成训练脚本"""
        script = '''#!/usr/bin/env python3
"""Qwen微调训练脚本"""
import argparse
import json
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
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    print(f"加载模型: {args.model_name}")
    model_path = {
        "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
        "Qwen2.5-1.8B": "Qwen/Qwen2.5-1.8B",
        "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    }.get(args.model_name, args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float32,
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

class MockXtuner:
    """模拟Xtuner命令行工具"""

    @staticmethod
    def run(config_path: str):
        print(f"调用Xtuner训练: {config_path}")
        pass
```

**Step 3: 创建 tests/test_finetuner.py**

```python
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
```

---

## 任务 5: 模型转换器模块

**Files:**
- Create: `src/converter/__init__.py`
- Create: `src/converter/converter.py`
- Create: `tests/test_converter.py`

**Step 1: 创建 src/converter/__init__.py**

```python
"""模型转换模块"""
from .converter import ModelConverter

__all__ = ["ModelConverter"]
```

**Step 2: 创建 src/converter/converter.py**

```python
"""模型转换器 - 将微调模型转换为Ollama格式"""
import os
import subprocess
from pathlib import Path
from typing import Optional

class ModelConverter:
    """模型转换器"""

    def __init__(self, base_model: str, adapter_path: str, output_dir: str = "./output"):
        self.base_model = base_model
        self.adapter_path = Path(adapter_path)
        self.output_dir = Path(output_dir)
        self.merged_dir = self.output_dir / "merged"

    def merge_adapter(self) -> str:
        """合并LoRA适配器到基础模型"""
        print("合并LoRA适配器到基础模型...")

        self.merged_dir.mkdir(parents=True, exist_ok=True)

        merge_script = self.output_dir / "merge.py"
        merge_script.write_text(f'''#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model = "{self.base_model}"
adapter_path = "{self.adapter_path}"
output_path = "{self.merged_dir}"

print(f"加载基础模型: {{base_model}}")
base_model_info = {{
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen2.5-1.8B": "Qwen/Qwen2.5-1.8B",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
}}.get(base_model, base_model)

tokenizer = AutoTokenizer.from_pretrained(base_model_info, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_info,
    device_map="cpu",
    torch_dtype=torch.float32,
    trust_remote_code=True
)

if Path(adapter_path).exists():
    from peft import PeftModel
    print(f"加载LoRA适配器: {{adapter_path}}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    print("适配器合并完成")

print(f"保存合并模型: {{output_path}}")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print("完成!")
''')

        subprocess.run(["python", str(merge_script)], check=True)
        return str(self.merged_dir)

    def export_to_gguf(self, output_path: Optional[str] = None) -> str:
        """导出为GGUF格式"""
        if output_path is None:
            output_path = self.output_dir / "model.gguf"
        else:
            output_path = Path(output_path)

        print(f"导出GGUF格式: {output_path}")

        export_script = self.merged_dir / "export.py"
        export_script.write_text(f'''#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import sys

model_path = "{self.merged_dir}"
output_path = "{output_path}"

print(f"加载模型: {{model_path}}")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")

print(f"导出GGUF到: {{output_path}}")
model.push_to_hub("tmp/quantized-model", safe_serialization=True)

print("请使用llama.cpp工具进行量化:")
print(f"  python -m llama_cpp翰 --model-type llama "
      f"-h {model_path} -o {output_path}")
''')

        return str(output_path)

    def create_ollama_model(self, model_name: str, quantize: str = "q4_0") -> str:
        """创建Ollama模型"""
        ollama_dir = self.output_dir / "ollama" / model_name
        ollama_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.merged_dir

        modelfile_content = f'''FROM {model_path}
TEMPLATE """{{{{ if .System }}}}
{{{{ .System }}}}
{{{{ end }}}}
{{{{ if .Prompt }}}}
{{{{ .Prompt }}}}
{{{{ end }}}}
"""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
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
                "--modelfile", str(modelfile_path)
            ], check=True, cwd=str(ollama_dir))
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("提示: Ollama未安装或不在PATH中，请手动运行:")
            print(f"  cd {ollama_dir}")
            print(f"  ollama create {model_name} -f Modelfile")

        return str(ollama_dir)

    def full_pipeline(self, model_name: str) -> str:
        """完整转换流程"""
        self.merge_adapter()
        ollama_dir = self.create_ollama_model(model_name)
        return ollama_dir
```

**Step 3: 创建 tests/test_converter.py**

```python
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
```

---

## 任务 6: CLI入口

**Files:**
- Create: `main.py`
- Create: `tests/test_cli.py`

**Step 1: 创建 main.py**

```python
#!/usr/bin/env python3
"""PDF微调程序 - 命令行入口"""
import sys
import click
from pathlib import Path

from src.doc_parser.parser import DocumentParser
from src.data_gen.generator import DataGenerator
from src.finetuner.trainer import Finetuner, FinetuneConfig
from src.converter.converter import ModelConverter


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """PDF微调程序 - 从PDF自动生成训练数据并微调模型"""
    pass


@cli.command()
@click.argument("pdf_dir", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", default="./data/docs.json", help="输出JSON路径")
@click.option("-r", "--recursive", is_flag=True, help="递归处理子目录")
def parse(doc_dir: str, output_path: str, recursive: bool):
    """解析文档"""
    click.echo(f"解析文档目录: {doc_dir}")

    parser = DocumentParser(doc_dir)
    documents = parser.parse_all(recursive=recursive)

    if documents:
        parser.save_to_json(documents, output_path)
        click.echo(f"成功解析 {len(documents)} 个PDF文件")
    else:
        click.echo("未找到任何PDF文件")
        sys.exit(1)


@cli.command()
@click.argument("docs_json", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", default="./data/train.jsonl", help="输出JSONL路径")
def generate(docs_json: str, output_path: str):
    """生成训练数据"""
    import json

    click.echo(f"从 {docs_json} 生成训练数据...")

    with open(docs_json, "r", encoding="utf-8") as f:
        documents = json.load(f)

    generator = DataGenerator()
    qa_list = generator.generate_from_documents(documents)

    generator.save_to_jsonl(qa_list, output_path)
    click.echo(f"生成 {len(qa_list)} 个问答对")


@cli.command()
@click.option("--model", "model_name", default="Qwen2.5-0.5B", help="模型名称")
@click.option("--data", "data_path", default="./data/train.jsonl", help="训练数据路径")
@click.option("-o", "--output", "output_dir", default="./output", help="输出目录")
@click.option("--epochs", "num_epochs", default=3, help="训练轮数")
@click.option("--batch-size", "batch_size", default=1, help="批次大小")
@click.option("--lr", "learning_rate", default=2e-4, help="学习率")
def finetune(model_name: str, data_path: str, output_dir: str, num_epochs: int, batch_size: int, learning_rate: float):
    """执行微调训练"""
    if not Path(data_path).exists():
        click.echo(f"错误: 训练数据不存在 {data_path}")
        click.echo("请先运行: python main.py generate <docs_json>")
        sys.exit(1)

    click.echo(f"开始微调: {model_name}")
    click.echo(f"训练数据: {data_path}")
    click.echo(f"输出目录: {output_dir}")

    config = FinetuneConfig(
        model_name=model_name,
        data_path=data_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    finetuner = Finetuner(config)
    finetuner.prepare_data()
    finetuner.train()

    click.echo(f"微调完成! 适配器保存在: {finetuner.get_adapter_path()}")


@cli.command()
@click.option("--base-model", "base_model", default="Qwen2.5-0.5B", help="基础模型")
@click.option("--adapter", "adapter_path", default="./output/adapter", help="LoRA适配器路径")
@click.option("-o", "--output", "output_dir", default="./output", help="输出目录")
@click.option("--name", "model_name", required=True, help="Ollama模型名称")
def convert(base_model: str, adapter_path: str, output_dir: str, model_name: str):
    """转换模型为Ollama格式"""
    click.echo(f"转换模型: {base_model} + {adapter_path}")

    converter = ModelConverter(
        base_model=base_model,
        adapter_path=adapter_path,
        output_dir=output_dir
    )

    ollama_dir = converter.full_pipeline(model_name)
    click.echo(f"Ollama模型已创建: {ollama_dir}")
    click.echo(f"运行: ollama run {model_name}")


@cli.command()
@click.option("--doc-dir", "doc_dir", required=True, help="文档目录 (支持PDF/DOCX/TXT/MD)")
@click.option("--model", "model_name", default="Qwen2.5-0.5B", help="模型名称")
@click.option("-o", "--output", "output_dir", default="./output", help="输出目录")
@click.option("--name", "model_name_ollama", "model_name", required=True, help="Ollama模型名称")
@click.option("--epochs", "num_epochs", default=3, help="训练轮数")
def all(doc_dir: str, model_name: str, output_dir: str, model_name_ollama: str, num_epochs: int):
    """一键执行全流程: 解析文档 -> 生成数据 -> 微调 -> 转换"""
    import tempfile

    click.echo("=== 文档微调程序 - 全流程 ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        docs_json = Path(tmpdir) / "docs.json"
        data_jsonl = Path(tmpdir) / "train.jsonl"

        click.echo("\n[1/4] 解析文档...")
        parser = DocumentParser(doc_dir)
        documents = parser.parse_all()
        parser.save_to_json(documents, str(docs_json))
        click.echo(f"  -> 解析完成: {len(documents)} 个文档")

        click.echo("\n[2/4] 生成训练数据...")
        with open(docs_json, "r", encoding="utf-8") as f:
            docs = json.load(f)
        generator = DataGenerator()
        qa_list = generator.generate_from_documents(docs)
        generator.save_to_jsonl(qa_list, str(data_jsonl))
        click.echo(f"  -> 生成完成: {len(qa_list)} 个问答对")

        click.echo("\n[3/4] 执行微调...")
        config = FinetuneConfig(
            model_name=model_name,
            data_path=str(data_jsonl),
            output_dir=output_dir,
            num_epochs=num_epochs,
        )
        finetuner = Finetuner(config)
        finetuner.prepare_data()
        finetuner.train()
        click.echo(f"  -> 微调完成: {finetuner.get_adapter_path()}")

        click.echo("\n[4/4] 转换Ollama模型...")
        converter = ModelConverter(
            base_model=model_name,
            adapter_path=finetuner.get_adapter_path(),
            output_dir=output_dir
        )
        ollama_dir = converter.full_pipeline(model_name_ollama)
        click.echo(f"  -> 转换完成: {ollama_dir}")

    click.echo("\n=== 完成 ===")
    click.echo(f"运行: ollama run {model_name_ollama}")


if __name__ == "__main__":
    cli()
```

---

## 任务 7: 创建测试目录和README

**Files:**
- Create: `tests/__init__.py`
- Create: `README.md`

**Step 1: 创建 tests/__init__.py**

```python
"""测试套件"""
```

**Step 2: 创建 README.md**

```markdown
# 文档微调程序

从文档(PDF/DOCX/TXT/MD)自动生成训练数据并微调模型，转换为Ollama格式。

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 解析文档

```bash
python main.py parse ./docs -o ./data/docs.json
# 支持格式: PDF, DOCX, TXT, Markdown
```

### 2. 生成训练数据

```bash
python main.py generate ./data/docs.json -o ./data/train.jsonl
```

### 3. 执行微调

```bash
python main.py finetune --model Qwen2.5-0.5B --data ./data/train.jsonl --output ./output --epochs 3
```

### 4. 转换Ollama格式

```bash
python main.py convert --adapter ./output/adapter --name my-model
```

### 5. 一键全流程

```bash
python main.py all --doc-dir ./docs --name my-model --epochs 3
```

## 模型支持

- Qwen2.5-0.5B (推荐，CPU友好，macOS/ Linux)
- Qwen2.5-1.8B
- Qwen2.5-7B

## 目录结构

```
.
├── main.py              # CLI入口
├── config.yaml           # 配置文件
├── requirements.txt     # 依赖
├── src/
│   ├── doc_parser/       # 文档解析
│   ├── data_gen/         # 数据生成
│   ├── finetuner/        # 微调
│   └── converter/        # 模型转换
├── data/                 # 数据目录
└── output/               # 输出目录
```
```

---

## 任务 8: 运行测试验证

**Step 1: 创建虚拟环境并安装依赖**

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install pytest
```

**Step 2: 运行测试**

```bash
pytest tests/ -v
```

**预期**: 所有测试通过

**Step 3: 创建示例PDF测试**

```bash
mkdir -p data/raw
# 放入一些PDF文件测试
python main.py parse data/raw -o data/docs.json
```

---

**计划完成时间**: 约2小时（取决于网络和硬件）
**复杂度**: 中等 - 多个独立模块，通过CLI集成
