#!/usr/bin/env python3
"""文档微调程序 - 命令行入口"""

import sys
import json
import click
from pathlib import Path

from src.doc_parser.parser import DocumentParser
from src.data_gen.generator import DataGenerator
from src.finetuner.trainer import Finetuner, FinetuneConfig
from src.converter.converter import ModelConverter


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """文档微调程序 - 从文档自动生成训练数据并微调模型"""
    pass


@cli.command()
def clean():
    """清理临时文件 (保留 output/adapter)"""
    import shutil

    removed = []

    if Path("output/temp").exists():
        shutil.rmtree("output/temp")
        removed.append("output/temp/")
        print("已清理: output/temp/")

    if not removed:
        click.echo("没有需要清理的临时文件")


@cli.command()
@click.argument("doc_dir", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "output_path",
    default="./output/temp/docs.json",
    help="输出JSON路径",
)
@click.option("-r", "--recursive", is_flag=True, help="递归处理子目录")
def parse(doc_dir: str, output_path: str, recursive: bool):
    """解析文档"""
    click.echo(f"解析文档目录: {doc_dir}")

    parser = DocumentParser(doc_dir)
    documents = parser.parse_all(recursive=recursive)

    if documents:
        parser.save_to_json(documents, output_path)
        click.echo(f"成功解析 {len(documents)} 个文档")
    else:
        click.echo("未找到任何文档")
        sys.exit(1)


@cli.command()
@click.argument("docs_json", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "output_path",
    default="./output/temp/train.jsonl",
    help="输出JSONL路径",
)
def generate(docs_json: str, output_path: str):
    """生成训练数据"""
    click.echo(f"从 {docs_json} 生成训练数据...")

    with open(docs_json, "r", encoding="utf-8") as f:
        documents = json.load(f)

    generator = DataGenerator()
    qa_list = generator.generate_from_documents(documents)

    generator.save_to_jsonl(qa_list, output_path)
    click.echo(f"生成 {len(qa_list)} 个问答对")


@cli.command()
@click.argument("docs_json", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "output_path",
    default="./output/temp/test.jsonl",
    help="输出JSONL路径",
)
@click.option(
    "--test-ratio",
    "test_ratio",
    default=0.2,
    help="测试集比例",
)
def generate_test(docs_json: str, output_path: str, test_ratio: float):
    """从文档生成测试集（与训练集分开）"""
    click.echo(f"从 {docs_json} 生成测试集 (比例: {test_ratio})...")

    with open(docs_json, "r", encoding="utf-8") as f:
        documents = json.load(f)

    generator = DataGenerator()
    qa_list = generator.generate_from_documents(documents)

    import random

    random.shuffle(qa_list)
    test_size = int(len(qa_list) * test_ratio)
    test_data = qa_list[:test_size]

    generator.save_to_jsonl(test_data, output_path)
    click.echo(f"生成 {len(test_data)} 个测试问答对")


@cli.command()
@click.option("--model", "model_name", default="Qwen2.5-0.5B", help="模型名称")
@click.option(
    "--model-path", "model_path", default=None, help="本地模型路径(优先于model_name)"
)
@click.option(
    "--data", "data_path", default="./output/temp/train.jsonl", help="训练数据路径"
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    default="./output",
    help="输出目录",
)
@click.option("--epochs", "num_epochs", default=3, help="训练轮数")
@click.option("--batch-size", "batch_size", default=1, help="批次大小")
@click.option("--lr", "learning_rate", default=2e-4, help="学习率")
def finetune(
    model_name: str,
    model_path: str,
    data_path: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
):
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
        model_path=model_path,
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
@click.option("--adapter", "adapter_path", default="./output/adapter", help="模型路径")
@click.option(
    "--data", "data_path", default="./output/temp/test.jsonl", help="测试数据路径"
)
@click.option("-o", "--output", "output_dir", default="./output", help="输出目录")
@click.option("--model", "model_name", default="Qwen2.5-0.5B", help="基础模型名称")
@click.option("--compare", is_flag=True, help="对比基础模型")
@click.option("--device", "device", default="cuda", help="设备: cuda/cpu")
def evaluate(
    adapter_path: str,
    data_path: str,
    output_dir: str,
    model_name: str,
    compare: bool,
    device: str,
):
    """评估微调后的模型"""
    if not Path(data_path).exists():
        click.echo(f"错误: 测试数据不存在 {data_path}")
        click.echo(f"提示: 先生成测试集: python main.py generate-test <docs_json>")
        sys.exit(1)

    if not Path(adapter_path).exists():
        click.echo(f"错误: 模型不存在 {adapter_path}")
        sys.exit(1)

    if compare:
        click.echo("\n" + "=" * 60)
        click.echo("【对比评估】")
        click.echo("=" * 60)

        click.echo("\n>>> 基础模型 (未训练)")
        click.echo("-" * 40)
        config_base = FinetuneConfig(
            model_name=model_name,
            model_path=None,
            data_path=data_path,
            output_dir=output_dir,
            device=device,
        )
        finetuner_base = Finetuner(config_base)
        finetuner_base.evaluate(data_path, use_base_model=True)

        click.echo("\n>>> 微调模型 (训练后)")
        click.echo("-" * 40)
        config = FinetuneConfig(
            model_name="local",
            model_path=adapter_path,
            data_path=data_path,
            output_dir=output_dir,
            device=device,
        )
        finetuner = Finetuner(config)
        finetuner.evaluate(data_path)

        click.echo("\n" + "=" * 60)
        click.echo("【对比完成】")
        click.echo("=" * 60)
    else:
        config = FinetuneConfig(
            model_name="local",
            model_path=adapter_path,
            data_path=data_path,
            output_dir=output_dir,
            device=device,
        )
        finetuner = Finetuner(config)
        result = finetuner.evaluate(data_path)
        click.echo(f"评估完成: 测试了 {result.get('test_count', 0)} 条数据")


@cli.command()
@click.option("--base-model", "base_model", default="Qwen2.5-0.5B", help="基础模型")
@click.option(
    "--base-model-path", "base_model_path", default=None, help="本地基础模型路径"
)
@click.option(
    "--adapter", "adapter_path", default="./output/adapter", help="LoRA适配器路径"
)
@click.option("-o", "--output", "output_dir", default="./output", help="输出目录")
@click.option("--name", "model_name", required=True, help="Ollama模型名称")
def convert(
    base_model: str,
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    model_name: str,
):
    """转换模型为Ollama格式"""
    click.echo(f"转换模型: {base_model} + {adapter_path}")

    base_model_for_convert = base_model_path if base_model_path else base_model
    converter = ModelConverter(
        base_model=base_model_for_convert,
        adapter_path=adapter_path,
        output_dir=output_dir,
    )

    ollama_dir = converter.full_pipeline(model_name)
    click.echo(f"Ollama模型已创建: {ollama_dir}")
    click.echo(f"运行: ollama run {model_name}")


@cli.command()
@click.option(
    "--doc-dir",
    "doc_dir",
    default="./train_docs",
    help="文档目录 (支持PDF/DOCX/TXT/MD)",
)
@click.option("--model", "model_name", default="Qwen2.5-0.5B", help="模型名称")
@click.option(
    "--model-path", "model_path", default=None, help="本地模型路径(优先于model_name)"
)
@click.option("-o", "--output", "output_dir", default="./output", help="输出目录")
@click.option("--name", "model_name_ollama", default="mymodel", help="Ollama模型名称")
@click.option("--epochs", "num_epochs", default=3, help="训练轮数")
@click.option("--test-ratio", "test_ratio", default=0.2, help="测试集比例")
@click.option("--device", "device", default="cuda", help="设备: cuda/cpu")
def all(
    doc_dir: str,
    model_name: str,
    model_path: str,
    output_dir: str,
    model_name_ollama: str,
    num_epochs: int,
    test_ratio: float,
    device: str,
):
    """一键执行全流程: 解析文档 -> 生成数据 -> 微调 -> 评估 -> 转换"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    docs_json = Path(output_dir) / "temp" / "docs.json"
    train_jsonl = Path(output_dir) / "temp" / "train.jsonl"
    test_jsonl = Path(output_dir) / "temp" / "test.jsonl"
    docs_json.parent.mkdir(parents=True, exist_ok=True)

    click.echo("=== 文档微调程序 - 全流程 ===")

    click.echo("\n[1/6] 解析文档...")
    parser = DocumentParser(doc_dir)
    documents = parser.parse_all(recursive=True)
    parser.save_to_json(documents, str(docs_json))
    click.echo(f"  -> 解析完成: {len(documents)} 个文档")

    click.echo("\n[2/6] 生成训练数据...")
    with open(docs_json, "r", encoding="utf-8") as f:
        docs = json.load(f)
    generator = DataGenerator()
    import random

    all_qa = generator.generate_from_documents(docs)
    random.shuffle(all_qa)
    test_size = int(len(all_qa) * test_ratio)
    train_qa = all_qa[test_size:]
    test_qa = all_qa[:test_size]
    generator.save_to_jsonl(train_qa, str(train_jsonl))
    generator.save_to_jsonl(test_qa, str(test_jsonl))
    click.echo(f"  -> 训练集: {len(train_qa)} 条, 测试集: {len(test_qa)} 条")

    click.echo("\n[3/6] 执行微调...")
    config = FinetuneConfig(
        model_name=model_name,
        model_path=model_path,
        data_path=str(train_jsonl),
        output_dir=output_dir,
        num_epochs=num_epochs,
        device=device,
    )
    finetuner = Finetuner(config)
    finetuner.prepare_data()
    finetuner.train()
    click.echo(f"  -> 微调完成: {finetuner.get_adapter_path()}")

    click.echo("\n[4/6] 评估模型...")
    result = finetuner.evaluate(str(test_jsonl))
    click.echo(f"  -> 评估完成")

    click.echo("\n[5/6] 转换Ollama模型...")
    converter = ModelConverter(
        base_model=model_path if model_path else model_name,
        adapter_path=finetuner.get_adapter_path(),
        output_dir=output_dir,
    )
    ollama_dir = converter.full_pipeline(model_name_ollama)
    click.echo(f"  -> 转换完成: {ollama_dir}")

    click.echo("\n[6/6] 清理临时文件...")
    if Path("output/temp").exists():
        import shutil

        for f in ["docs.json", "train.jsonl", "test.jsonl", "train.py", "evaluate.py"]:
            p = Path("output/temp") / f
            if p.exists():
                p.unlink()
        if (Path("output/temp") / "data").exists():
            shutil.rmtree(Path("output/temp") / "data")
        if (Path("output/temp") / "__pycache__").exists():
            shutil.rmtree(Path("output/temp") / "__pycache__")
    click.echo("  -> 清理完成")

    click.echo("\n=== 完成 ===")
    click.echo(f"运行: ollama run {model_name_ollama}")


if __name__ == "__main__":
    cli()
