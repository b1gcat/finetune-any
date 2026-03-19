# 文档微调程序设计文档

**日期**: 2026-03-19  
**项目**: finetune-any

## 1. 背景与目标

### 1.1 需求概述
开发一个命令行微调程序，实现以下流程：
1. 读取目录下所有文档（PDF/DOCX/TXT/Markdown）
2. 自动生成训练数据（基于标准/规定文档）
3. 使用CPU启动微调（支持后期扩展GPU）
4. 生成微调模型并转换为Ollama格式

### 1.2 技术选型

| 组件 | 选型 | 说明 |
|-----|------|------|
| 基础模型 | Qwen2.5-0.5B | 开源可商用，CPU友好 |
| 微调框架 | Xtuner | 轻量级，简单易用 |
| 文档解析 | pdfplumber + python-docx + 内置 | 多格式支持 |
| 训练数据格式 | Alpaca JSONL | 通用格式 |
| 模型转换 | llama.cpp | 转gguf格式 |
| Ollama支持 | ollama create | 官方工具 |

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      CLI 入口                           │
│                   (main.py / cli.py)                    │
└─────────────────────┬───────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ 文档解析器  │ │ 数据生成器  │ │ 微调管理器  │
│ doc_parser │ │ data_gen    │ │ finetuner   │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │
       ▼               ▼               ▼
  [文本提取]      [训练数据]       [LoRA权重]
                   (JSONL)            │
                                     ▼
                            ┌─────────────┐
                            │ 模型转换器  │
                            │ converter   │
                            └──────┬──────┘
                                   │
                                   ▼
                            [Ollama模型]
```

## 3. 模块设计

### 3.1 文档解析器 (`pdf_parser/`)

**功能**: 读取目录下所有文档（PDF/DOCX/TXT/Markdown），提取文本内容

```python
# 主要接口
class DocumentParser:
    def __init__(self, doc_dir: str)
    def parse_all(self) -> List[Document]
    def parse_single(self, doc_path: str) -> Document
```

**数据结构**:
```python
@dataclass
class Document:
    filename: str
    pages: List[Page]
    metadata: Dict  # 标题、作者、页数等

@dataclass
class Page:
    page_num: int
    text: str
    tables: List[Table]  # 可选的表格数据
```

**依赖**: pdfplumber

### 3.2 训练数据生成器 (`data_gen/`)

**功能**: 从标准/规定文本生成问答训练数据

**生成策略**:
1. **规则匹配** - 识别条款编号（第一章、第X条、1.1等）
2. **关键实体提取** - 人名、时间、地点、金额等
3. **问答模板** - 针对定义、要求、禁止、义务等类型生成问答

**训练数据格式** (Alpaca):
```json
{
  "instruction": "根据以下标准回答问题",
  "input": "《XXX标准》规定了什么内容？",
  "output": "《XXX标准》主要规定了..."
}
```

**主要接口**:
```python
class DataGenerator:
    def __init__(self, documents: List[Document])
    def generate_qa_pairs(self) -> List[QA]
    def save_to_jsonl(self, output_path: str)
```

### 3.3 微调管理器 (`finetuner/`)

**功能**: 调用Xtuner执行微调

**流程**:
1. 检查/下载基础模型
2. 准备Xtuner配置文件
3. 执行微调训练
4. 保存LoRA权重

**配置参数**:
```yaml
# 默认配置
model_name_or_path: Qwen2.5-0.5B
data_path: ./data/train.jsonl
max_length: 2048
batch_size: 1
learning_rate: 2e-4
num_epochs: 3
optimizer_type: adamw_torch
accumulation_steps: 16
```

**主要接口**:
```python
class Finetuner:
    def __init__(self, config: FinetuneConfig)
    def prepare_model(self)
    def train(self)
    def save_adapter(self, output_path: str)
```

### 3.4 模型转换器 (`converter/`)

**功能**: 将微调后的模型转换为Ollama格式

**流程**:
1. 合并LoRA权重到基础模型
2. 量化模型（Q4_K_M）
3. 生成Modelfile
4. 创建Ollama模型

**主要接口**:
```python
class ModelConverter:
    def __init__(self, base_model: str, adapter_path: str)
    def merge_and_export(self, output_dir: str)
    def create_ollama_model(self, model_name: str)
```

## 4. CLI设计

### 4.1 命令结构

```bash
python main.py [OPTIONS] COMMAND [ARGS]...
```

### 4.2 子命令

#### 4.2.1 parse - 解析PDF
```bash
python main.py parse ./pdf_dir -o ./data/docs.json
```

#### 4.2.2 generate - 生成训练数据
```bash
python main.py generate ./data/docs.json -o ./data/train.jsonl
```

#### 4.2.3 finetune - 执行微调
```bash
python main.py finetune \
    --model Qwen2.5-0.5B \
    --data ./data/train.jsonl \
    --output ./output \
    --epochs 3 \
    --batch-size 1
```

#### 4.2.4 convert - 转换Ollama格式
```bash
python main.py convert \
    --base-model ./output/base_model \
    --adapter ./output/adapter \
    --model-name my-finetuned-model
```

#### 4.2.5 all - 一键执行全流程
```bash
python main.py all \
    --pdf-dir ./pdf_dir \
    --model Qwen2.5-0.5B \
    --output ./output \
    --model-name my-model
```

### 4.3 全局参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| --config | 配置文件路径 | config.yaml |
| --verbose | 详细输出 | False |
| --dry-run | 仅模拟运行 | False |

## 5. 目录结构

```
finetune-any/
├── main.py                 # CLI入口
├── config.yaml             # 默认配置
├── requirements.txt       # Python依赖
├── README.md
├── src/
│   ├── doc_parser/         # 文档解析模块
│   ├── data_gen/           # 数据生成模块
│   ├── finetuner/          # 微调模块
│   └── converter/          # 模型转换模块
│       ├── __init__.py
│       └── converter.py
├── data/                   # 数据目录
│   ├── raw/                # 原始PDF
│   ├── processed/          # 处理后数据
│   └── train.jsonl         # 训练数据
└── output/                 # 输出目录
    ├── adapter/            # LoRA权重
    ├── merged/             # 合并后模型
    └── ollama/             # Ollama模型
```

## 6. 依赖清单

```
# requirements.txt
pdfplumber>=0.10.0          # PDF解析
python-docx>=0.8.0          # Word文档解析
markdown>=3.4.0             # Markdown解析
beautifulsoup4>=4.12.0     # HTML解析 (Markdown依赖)
lxml>=4.9.0                # XML/HTML解析
xtuner>=0.1.0               # 微调框架 (通过pip安装)
transformers>=4.35.0        # 模型加载
torch>=2.0.0                # PyTorch (CPU版本)
accelerate>=0.20.0          # 加速训练
bitsandbytes>=0.37.0         # 量化支持
safetensors>=0.3.0          # 安全张量格式
llama-cpp-python>=0.2.0     # 模型转换
click>=8.0                  # CLI框架
pyyaml>=6.0                 # 配置文件
tqdm>=4.65                  # 进度条
```

## 7. 错误处理

| 场景 | 处理方式 |
|-----|---------|
| PDF解析失败 | 跳过该文件，记录日志，继续处理其他文件 |
| 训练数据为空 | 报错退出，提示检查PDF内容 |
| 磁盘空间不足 | 提前检查，报错退出 |
| 模型下载失败 | 重试3次，超时后报错 |
| 微调中断 | 支持断点续训，保存checkpoint |

## 8. 后续扩展

### 8.1 GPU支持
- 添加 `--device cuda` 参数
- 自动检测CUDA可用性
- 支持多卡训练 (DeepSpeed)

### 8.2 其他模型支持
- Llama3-8B
- ChatGLM3-6B
- Phi-3

### 8.3 其他数据格式
- Word文档 (.docx)
- Markdown
- 网页抓取

### 8.4 其他微调框架
- Unsloth (更快)
- LLaMA-Factory (更丰富)
