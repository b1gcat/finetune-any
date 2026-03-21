# 微调程序实施计划 (v2)

> **需求变更说明**: 移除文档解析功能，改为使用预置数据集进行微调。

**Goal:** 实现一个交互式微调程序，加载预置数据集→微调→评估→转Ollama格式

**Architecture:** 采用模块化设计，分为数据加载、微调管理、模型转换三个核心模块，通过交互式CLI统一入口，支持训练、评估、转换三种模式。

**Tech Stack:** Python 3.10+, transformers, torch, llama-cpp-python, Click

---

## 主要变更

### 已移除
- `src/doc_parser/` - 文档解析模块
- 文档解析相关命令 (parse, generate, generate-test, all)

### 新增功能
- 交互式 CLI 引导
- 三种操作模式: 完整流程 / 仅评估 / 仅转换
- 数据集目录支持 (`dataset/` 目录)
- 随机 20% 抽样评估

---

## 任务 1: 项目初始化

**目录结构**:
```
finetune-any/
├── main.py                 # 交互式CLI入口
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖
├── src/
│   ├── data_gen/           # 数据加载模块
│   ├── finetuner/          # 微调模块
│   └── converter/          # 模型转换模块
├── dataset/                 # 数据集目录 (train.jsonl, test.jsonl)
├── output/                  # 输出目录
│   ├── adapter/            # LoRA权重
│   ├── merged/             # 合并后模型
│   └── ollama/             # Ollama模型
└── docs/plans/             # 计划文档
```

---

## 任务 2: 数据加载模块 (已更新)

**Files:**
- `src/data_gen/generator.py` - 已简化

**功能**:
1. 从 `dataset/` 目录加载 `train.jsonl` 和 `test.jsonl`
2. 支持从训练集自动分割测试集 (按比例)
3. 支持随机 20% 抽样评估

**数据格式** (Alpaca JSONL):
```json
{"instruction": "根据以下文档内容回答问题", "input": "问题内容", "output": "回答内容"}
```

**主要接口**:
```python
class DataGenerator:
    def load_from_jsonl(self, jsonl_path: str) -> List[QA]
    def load_dataset_dir(self, dataset_dir: str, test_ratio: float) -> tuple
    def save_to_jsonl(self, qa_list: List[QA], output_path: str)
```

---

## 任务 3: 交互式 CLI

**Files:**
- `main.py` - 重写为交互式入口

**交互流程**:

```
╔═══════════════════════════════════════════════════════╗
║           微调程序 - 交互式引导                        ║
╚═══════════════════════════════════════════════════════╝

请选择操作模式:
  1. 完整流程 (训练 + 评估 + 转换)
  2. 仅评估 (需要已有训练好的模型)
  3. 仅转换 (需要已有 adapter)

[1/5] 选择数据集
  1. 从 ./dataset 目录加载
  2. 指定自定义目录
  3. 直接指定 jsonl 文件

[2/5] 选择基础模型
  1. Qwen2.5-0.5B (最小，最快)
  2. Qwen2.5-1.8B (中等)
  3. Qwen2.5-7B (较大，效果更好)
  4. 本地模型

[3/5] 设置训练参数
  训练轮数 [默认: 3]:
  测试集比例 [默认: 0.2]:
  设备选择: 1. CUDA (GPU)  2. CPU

[4/5] 设置输出
  输出目录 [默认: ./output]:
  Ollama 模型名称 [默认: mymodel]:

[5/5] 确认配置
  (显示配置汇总)
  确认开始训练? [Y/n]:
```

**执行流程**:

| 模式 | 步骤 |
|-----|------|
| 完整流程 | 1.加载数据集 → 2.执行微调 → 3.随机20%评估 → 4.转换Ollama |
| 仅评估 | 1.选择adapter → 2.随机20%抽样评估 |
| 仅转换 | 1.选择adapter → 2.转换Ollama |

---

## 任务 4: 评估逻辑 (新增)

**评估策略**:
1. 优先使用 `test.jsonl` (如存在)
2. 若无测试集，随机从 `train.jsonl` 抽取 **20%** 进行评估
3. 评估模式同样采用随机 20% 抽样

**实现**:
```python
def get_evaluation_data(dataset_dir, test_ratio=0.2):
    # 加载全部数据
    # 随机打乱
    # 抽取20%作为评估集
    return train_data, eval_data
```

---

## 任务 5: 微调管理器模块

**Files:**
- `src/finetuner/trainer.py`

**功能**:
1. 检查/下载基础模型
2. 准备训练数据
3. 执行微调训练 (LoRA)
4. 评估模型
5. 保存LoRA权重

**配置参数**:
```python
@dataclass
class FinetuneConfig:
    model_name: str = "Qwen2.5-0.5B"
    model_path: Optional[str] = None
    data_path: str = "./output/temp/train.jsonl"
    output_dir: str = "./output"
    max_length: int = 2048
    batch_size: int = 1
    learning_rate: float = 2e-4
    num_epochs: int = 3
    device: str = "cuda"
```

---

## 任务 6: 模型转换器模块

**Files:**
- `src/converter/converter.py`

**功能**:
1. 合并LoRA权重到基础模型
2. 生成Modelfile
3. 创建Ollama模型

**流程**:
```
adapter + base_model → merged → ollama/{model_name}/
```

---

## 使用方法

### 准备数据集

```bash
# 方式1: 将数据集放入 dataset 目录
mv my_train.jsonl dataset/train.jsonl

# 方式2: 自定义目录
mv my_data.jsonl /path/to/dataset/train.jsonl
```

### 运行程序

```bash
# 直接运行，交互式引导
python main.py
```

### 程序输出

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

运行命令: ollama run mymodel
```

---

## 目录结构 (最终)

```
finetune-any/
├── main.py                 # 交互式CLI入口
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖
├── src/
│   ├── data_gen/           # 数据加载模块
│   │   ├── __init__.py
│   │   └── generator.py
│   ├── finetuner/          # 微调模块
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── converter/          # 模型转换模块
│       ├── __init__.py
│       └── converter.py
├── dataset/                 # 数据集目录
│   ├── train.jsonl         # 训练数据
│   └── test.jsonl          # 测试数据 (可选)
├── output/                  # 输出目录
│   ├── temp/              # 临时文件
│   ├── adapter/           # LoRA权重
│   ├── merged/           # 合并后模型
│   └── ollama/           # Ollama模型
└── docs/
    └── plans/             # 计划文档
```

---

**计划完成时间**: 约1小时（代码已实现）
**复杂度**: 低 - 简化后只需数据加载和训练功能
