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
@click.argument("doc_dir", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", default="./data/docs.json", help="输出JSON路径")
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
@click.option("-o", "--output", "output_path", default="./data/train.jsonl", help="输出JSONL路径")
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
@click.option("--name", "model_name_ollama", required=True, help="Ollama模型名称")
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
