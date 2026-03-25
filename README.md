# 文档微调程序

从文档(PDF/DOCX/TXT/MD)自动生成训练数据并微调模型，转换为Ollama格式。

## 安装

```bash
# 安装依赖
uv pip install -r requirements.txt
uv pip install pdfplumber  # PDF解析
```

## 快速开始

```bash
# 一键全流程（使用默认参数）
uv run main.py all

# 指定参数
uv run main.py all --doc-dir ./train_docs --model Qwen2.5-0.5B --name mymodel --epochs 1 --device cuda
```

## 命令说明

### all - 一键全流程
```bash
uv run main.py all [选项]
```
- 流程: 解析文档 -> 生成训练/测试集 -> 微调训练 -> 评估 -> 转换Ollama -> 清理
- 默认参数:
  - `--doc-dir ./train_docs` - 文档目录
  - `--name mymodel` - Ollama模型名称
  - `--model Qwen2.5-0.5B` - 模型名称
  - `--epochs 3` - 训练轮数
  - `--test-ratio 0.2` - 测试集比例
  - `--device cuda` - 设备 (cuda/cpu)

### parse - 解析文档
```bash
uv run main.py parse ./train_docs -o ./output/temp/docs.json -r
```
- 支持格式: PDF, DOCX, TXT, Markdown
- `-r`: 递归处理子目录

### generate - 生成训练数据
```bash
uv run main.py generate ./output/temp/docs.json -o ./output/temp/train.jsonl
```

### generate-test - 生成测试集
```bash
uv run main.py generate-test ./output/temp/docs.json -o ./output/temp/test.jsonl --test-ratio 0.2
```

### finetune - 微调训练
```bash
uv run main.py finetune --model Qwen2.5-0.5B -o ./output --epochs 1 --device cuda
```
- 模型不存在时自动从 ModelScope 下载
- `--model-path`: 使用本地模型

### evaluate - 评估模型
```bash
# 评估微调模型（使用测试集）
uv run main.py evaluate

# 指定测试数据
uv run main.py evaluate --data ./output/temp/test.jsonl

# 指定设备
uv run main.py evaluate --device cuda

# 对比评估（基础模型 vs 微调模型）
uv run main.py evaluate --compare --device cuda
```
- 评估结果保存到: `output/temp/test_eval_result.jsonl`

### convert - 转换为 Ollama 格式
```bash
uv run main.py convert --adapter ./output/adapter --name mymodel
```

### clean - 清理临时文件
```bash
uv run main.py clean
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--doc-dir` | `./train_docs` | 文档目录 |
| `--model` | `qwen3:0.6b` | 模型名称 |
| `--name` | `mymodel` | Ollama模型名称 |
| `--epochs` | `3` | 训练轮数 |
| `--test-ratio` | `0.2` | 测试集比例 |
| `--device` | `cuda` | 设备 (cuda/cpu) |

## 模型支持

- **qwen3:0.6b** (默认，当前仅支持小模型)

## 设备配置

| 设备 | max_length | 说明 |
|------|------------|------|
| CUDA (GPU) | 2048 | 推荐使用GPU加速 |
| CPU | 512 | 最小内存配置 |

## Ollama 参数

转换后的模型默认使用以下参数（适合报告审核场景）：

```dockerfile
PARAMETER temperature 0.1   # 低温度，稳定准确
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER repeat_penalty 1.1
```

## 目录结构

```
.
├── main.py              # CLI入口
├── requirements.txt     # 依赖
├── train_docs/          # 源文件目录 (PDF/DOCX等)
└── output/
    ├── temp/           # 临时文件 (可清理)
    │   ├── docs.json       # 解析后的文档
    │   ├── train.jsonl     # 训练数据
    │   ├── test.jsonl     # 测试数据
    │   └── test_eval_result.jsonl  # 评估结果
    └── adapter/        # 微调后的模型
```

## 模型下载

模型默认从 ModelScope 下载，保存到 `~/.cache/modelscope/`