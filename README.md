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
