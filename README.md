# 文档微调程序

从文档(PDF/DOCX/TXT/MD)自动生成训练数据并微调模型，转换为Ollama格式。

## 安装

```bash
pip install -r requirements.txt
pip install pdfplumber  # PDF解析
```

## 快速开始

```bash
# 一键全流程
python main.py all --doc-dir ./train_docs --model Qwen2.5-0.5B --name mymodel --epochs 1
```

## 命令说明

### parse - 解析文档
```bash
python main.py parse ./train_docs -o ./output/temp/docs.json -r
```
- 支持格式: PDF, DOCX, TXT, Markdown
- `-r`: 递归处理子目录

### generate - 生成训练数据
```bash
python main.py generate ./output/temp/docs.json -o ./output/temp/train.jsonl
```

### finetune - 微调训练
```bash
python main.py finetune --model Qwen2.5-0.5B -o ./output --epochs 1
```
- 模型不存在时自动从 ModelScope 下载
- `--model-path`: 使用本地模型

### convert - 转换为 Ollama 格式
```bash
python main.py convert --adapter ./output/adapter --name mymodel
```

### clean - 清理临时文件
```bash
python main.py clean  # 清理 temp/ 目录
```

## 模型支持

- Qwen2.5-0.5B (推荐，CPU友好)
- Qwen2.5-1.8B
- Qwen2.5-7B

## 目录结构

```
.
├── main.py              # CLI入口
├── requirements.txt     # 依赖
├── train_docs/          # 源文件目录 (PDF/DOCX等)
└── output/
    ├── temp/           # 临时文件 (可清理)
    │   ├── docs.json
    │   └── train.jsonl
    └── adapter/        # 微调后的模型
```

## 模型下载

模型默认从 ModelScope 下载，保存到 `~/.cache/modelscope/`
