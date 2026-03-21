# 微调程序设计文档 (v2)

**日期**: 2026-03-21  
**项目**: finetune-any  
**版本**: v2 (简化版)

## 1. 变更概述

### 1.1 主要变更

| 变更项 | 旧版 | 新版 |
|-------|-----|-----|
| 数据来源 | PDF/DOCX文档解析 | 预置JSONL数据集 |
| CLI方式 | Click命令行参数 | 交互式引导 |
| 操作模式 | 仅完整流程 | 三种模式(训练/评估/转换) |
| 评估抽样 | 前5条快速评估 | 随机20%抽样 |

### 1.2 移除的模块

- `src/doc_parser/` - 文档解析模块已完全移除

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                   交互式 CLI 入口                       │
│                      (main.py)                          │
└─────────────────────┬───────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
   ┌───────────┐ ┌───────────┐ ┌───────────┐
   │ 数据加载器 │ │ 微调管理器│ │ 模型转换器│
   │ data_gen  │ │ finetuner │ │ converter │
   └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
         │             │             │
         ▼             ▼             ▼
   [dataset/]    [LoRA权重]    [Ollama模型]
   train.jsonl   adapter/      ollama/{name}/
   test.jsonl
```

## 3. 模块设计

### 3.1 数据加载器 (`data_gen/`)

**功能**: 从数据集目录加载训练/测试数据

**主要接口**:
```python
class DataGenerator:
    def load_from_jsonl(self, jsonl_path: str) -> List[QA]
    def load_dataset_dir(self, dataset_dir: str, test_ratio: float) -> tuple
    def save_to_jsonl(self, qa_list: List[QA], output_path: str)
```

**数据结构**:
```python
@dataclass
class QA:
    instruction: str  # 指令
    input: str        # 问题
    output: str       # 回答
```

### 3.2 微调管理器 (`finetuner/`)

**功能**: 执行模型微调训练

**流程**:
1. 检查/下载基础模型 (Qwen2.5 系列)
2. 准备训练数据
3. 执行 LoRA 微调
4. 保存 LoRA 权重到 `output/adapter/`

**评估流程**:
1. 若有 `test.jsonl`，使用测试数据
2. 若无，随机从训练集抽取 20% 作为评估样本
3. 执行推理并输出结果

### 3.3 模型转换器 (`converter/`)

**功能**: 合并 LoRA 并转换为 Ollama 格式

**流程**:
```
base_model + adapter → merge → ollama/{model_name}/
```

## 4. 交互式 CLI 设计

### 4.1 操作模式选择

```
╔═══════════════════════════════════════════════════════╗
║           微调程序 - 交互式引导                        ║
╚═══════════════════════════════════════════════════════╝

请选择操作模式:
  1. 完整流程 (训练 + 评估 + 转换)
  2. 仅评估 (需要已有训练好的模型)
  3. 仅转换 (需要已有 adapter)
```

### 4.2 配置步骤

| 步骤 | 内容 | 选项 |
|-----|------|-----|
| 1 | 选择数据集 | dataset目录 / 自定义目录 / jsonl文件 |
| 2 | 选择模型 | Qwen2.5-0.5B/1.8B/7B / 本地模型 |
| 3 | 训练参数 | 轮数、测试集比例、设备 |
| 4 | 输出设置 | 输出目录、Ollama名称 |
| 5 | 确认执行 | 配置汇总、确认 |

### 4.3 执行状态

```
[1/4] 加载数据集: ./dataset
  ✓ 训练集: 800 条, 测试集: 200 条

[2/4] 执行微调...
  ✓ 微调完成: ./output/adapter

[3/4] 评估模型...
  使用 200 条测试数据进行评估...
  ✓ 评估完成

[4/4] 转换 Ollama 模型...
  ✓ Ollama 模型已创建: ./output/ollama/mymodel
```

## 5. 数据格式

### 5.1 训练数据格式 (Alpaca JSONL)

```json
{"instruction": "根据以下文档内容回答问题", "input": "问题内容", "output": "回答内容"}
{"instruction": "根据以下文档内容回答问题", "input": "问题内容", "output": "回答内容"}
```

### 5.2 数据集目录结构

```
dataset/
├── train.jsonl         # 训练数据 (必需)
└── test.jsonl          # 测试数据 (可选)
```

## 6. 目录结构

```
finetune-any/
├── main.py                 # 交互式CLI入口
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖
├── src/
│   ├── data_gen/           # 数据加载模块
│   ├── finetuner/          # 微调模块
│   └── converter/          # 模型转换模块
├── dataset/                 # 数据集目录
│   ├── train.jsonl
│   └── test.jsonl
└── output/                  # 输出目录
    ├── temp/              # 临时文件
    ├── adapter/           # LoRA权重
    ├── merged/            # 合并后模型
    └── ollama/            # Ollama模型
```

## 7. 依赖清单

```
transformers>=4.35.0        # 模型加载
torch>=2.0.0                # PyTorch
accelerate>=0.20.0          # 加速训练
peft>=0.5.0                 # LoRA支持
llama-cpp-python>=0.2.0     # 模型转换
ollama                       # Ollama CLI
```

## 8. 支持的模型

| 模型 | 参数量 | 说明 |
|-----|-------|-----|
| Qwen2.5-0.5B | 0.5B | 最小，推荐测试用 |
| Qwen2.5-1.8B | 1.8B | 中等大小 |
| Qwen2.5-7B | 7B | 较大，效果更好 |
| 本地模型 | - | 支持加载本地路径 |

## 9. 错误处理

| 场景 | 处理方式 |
|-----|---------|
| 数据集目录不存在 | 创建目录，提示用户放入数据 |
| 训练数据为空 | 报错退出 |
| 模型下载失败 | 重试后报错 |
| 微调中断 | 保存 checkpoint |
| Ollama 未安装 | 提示手动运行 |

## 10. 后续扩展

- [ ] 支持更多模型 (Llama3, ChatGLM)
- [ ] 支持增量训练
- [ ] 支持量化配置
- [ ] 支持断点续训
