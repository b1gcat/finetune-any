#!/usr/bin/env python3
"""微调程序 - 交互式引导"""

import os
import sys
from pathlib import Path

from src.data_gen.generator import DataGenerator
from src.finetuner.trainer import Finetuner, FinetuneConfig
from src.converter.converter import ModelConverter


def print_banner():
    banner = """
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║           微调程序 - 交互式引导                        ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
    """
    print(banner)


def print_step(step, total, title):
    print(f"\n{'='*50}")
    print(f"  步骤 {step}/{total}: {title}")
    print(f"{'='*50}")


def input_choice(prompt, options, default=None):
    """显示选项并获取用户选择"""
    while True:
        print(f"\n{prompt}")
        for i, opt in enumerate(options, 1):
            marker = " ← 默认" if default and opt == default else ""
            print(f"  {i}. {opt}{marker}")
        
        choice = input("\n请选择 (直接回车使用默认): ").strip()
        
        if not choice and default:
            return default
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        
        print("无效选择，请重新输入")


def input_text(prompt, default=None, required=True):
    """获取文本输入"""
    while True:
        default_str = f" (直接回车: {default})" if default else ""
        value = input(f"\n{prompt}{default_str}: ").strip()
        
        if not value and default:
            return default
        
        if not value and required:
            print("此项为必填，请输入")
            continue
        
        return value


def input_yes_no(prompt, default=True):
    """是/否确认"""
    default_str = " [Y/n]" if default else " [y/N]"
    while True:
        value = input(f"\n{prompt}{default_str}: ").strip().lower()
        
        if not value:
            return default
        
        if value in ['y', 'yes', '是']:
            return True
        elif value in ['n', 'no', '否']:
            return False
        
        print("请输入 y 或 n")


def main():
    print_banner()
    
    config = {
        'mode': None,  # 'train', 'evaluate', 'convert'
        'dataset_dir': None,
        'train_data': None,
        'test_data': None,
        'model_name': None,
        'model_path': None,
        'adapter_path': None,
        'output_dir': './output',
        'ollama_name': None,
        'num_epochs': 3,
        'test_ratio': 0.2,
        'device': 'cuda',
    }
    
    print("\n请选择操作模式:")
    print("  1. 完整流程 (训练 + 评估 + 转换)")
    print("  2. 仅评估 (需要已有训练好的模型)")
    print("  3. 仅转换 (需要已有 adapter)")
    
    mode_choice = input("\n请选择 (1-3): ").strip()
    
    if mode_choice == "2":
        config['mode'] = 'evaluate'
    elif mode_choice == "3":
        config['mode'] = 'convert'
    else:
        config['mode'] = 'train'
    
    print("\n开始配置...")
    
    # 评估和转换模式需要的数据
    if config['mode'] in ('evaluate', 'convert'):
        print("\n请选择数据集来源 (用于加载测试数据或配置):")
        print("  1. 从 ./dataset 目录加载")
        print("  2. 指定自定义目录")
        
        dataset_choice = input("\n请选择 (1-2): ").strip()
        if dataset_choice == "2":
            config['dataset_dir'] = input_text("请输入数据集目录路径", required=False)
        else:
            config['dataset_dir'] = './dataset'
    
    # 步骤 2: 选择模型 (所有模式都需要)
    print_step(2 if config['mode'] == 'train' else 2, 5 if config['mode'] == 'train' else 2, "选择基础模型")
    
    # 步骤 1: 选择数据集
    print_step(1, 5, "选择数据集")
    
    print("\n请选择数据集来源:")
    print("  1. 从 ./dataset 目录加载")
    print("  2. 指定自定义目录")
    print("  3. 直接指定 jsonl 文件")
    
    dataset_choice = input("\n请选择 (1-3): ").strip()
    
    if dataset_choice == "1":
        config['dataset_dir'] = './dataset'
    elif dataset_choice == "2":
        config['dataset_dir'] = input_text("请输入数据集目录路径", required=True)
    elif dataset_choice == "3":
        config['train_data'] = input_text("请输入训练数据文件路径 (train.jsonl)", required=True)
        config['test_data'] = input_text("请输入测试数据文件路径 (test.jsonl，可选)", required=False) or None
    else:
        config['dataset_dir'] = './dataset'
    
    # 检查数据集是否存在
    if config['dataset_dir']:
        if not Path(config['dataset_dir']).exists():
            print(f"\n✗ 目录不存在: {config['dataset_dir']}")
            print("  将创建目录，请在其中放置 train.jsonl 文件")
            Path(config['dataset_dir']).mkdir(parents=True, exist_ok=True)
    
    # 步骤 2: 选择模型
    print_step(2, 5, "选择基础模型")
    
    print("\n请选择基础模型:")
    print("  1. Qwen2.5-0.5B (最小，最快，适合测试)")
    print("  2. Qwen2.5-1.8B (中等大小)")
    print("  3. Qwen2.5-7B (较大，效果更好)")
    print("  4. 本地模型")
    
    model_choice = input("\n请选择 (1-4): ").strip()
    
    if model_choice == "1":
        config['model_name'] = "Qwen2.5-0.5B"
    elif model_choice == "2":
        config['model_name'] = "Qwen2.5-1.8B"
    elif model_choice == "3":
        config['model_name'] = "Qwen2.5-7B"
    elif model_choice == "4":
        config['model_path'] = input_text("请输入本地模型路径", required=True)
        config['model_name'] = "local"
    else:
        config['model_name'] = "Qwen2.5-0.5B"
    
    # 步骤 3: 训练参数
    print_step(3, 5, "设置训练参数")
    
    try:
        epochs = int(input("\n训练轮数 [默认: 3]: ").strip() or "3")
    except ValueError:
        epochs = 3
    config['num_epochs'] = epochs
    
    try:
        test_ratio = float(input("测试集比例 [默认: 0.2]: ").strip() or "0.2")
    except ValueError:
        test_ratio = 0.2
    config['test_ratio'] = test_ratio
    
    print("\n设备选择:")
    print("  1. CUDA (GPU)")
    print("  2. CPU")
    device_choice = input("请选择 (1-2) [默认: 1]: ").strip()
    config['device'] = 'cuda' if device_choice != "2" else 'cpu'
    
    # 步骤 4: 输出设置
    print_step(4, 5, "设置输出")
    
    config['output_dir'] = input_text("输出目录", default='./output', required=False)
    config['ollama_name'] = input_text("Ollama 模型名称", default='mymodel', required=False)
    
    # 步骤 5: 确认并执行
    print_step(5, 5, "确认配置")
    
    print("\n" + "=" * 50)
    print("  配置汇总")
    print("=" * 50)
    print(f"  操作模式: ", end="")
    if config['mode'] == 'train':
        print("完整流程")
    elif config['mode'] == 'evaluate':
        print("仅评估")
    else:
        print("仅转换")
    
    if config['mode'] in ('train', 'evaluate'):
        if config['dataset_dir']:
            print(f"  数据集目录: {config['dataset_dir']}")
        else:
            print(f"  训练数据: {config['train_data']}")
            if config['test_data']:
                print(f"  测试数据: {config['test_data']}")
    
    print(f"  基础模型: {config['model_name']}")
    if config['model_path']:
        print(f"  模型路径: {config['model_path']}")
    
    if config['mode'] == 'train':
        print(f"  训练轮数: {config['num_epochs']}")
        print(f"  测试集比例: {config['test_ratio']}")
        print(f"  设备: {config['device']}")
    
    print(f"  输出目录: {config['output_dir']}")
    
    if config['mode'] in ('train', 'convert'):
        print(f"  Ollama 名称: {config['ollama_name']}")
    
    if config['mode'] == 'evaluate':
        print(f"  Adapter 路径: {config['adapter_path']}")
    
    print("=" * 50)
    
    mode_text = "开始" if config['mode'] == 'train' else "继续"
    if not input_yes_no(f"确认{mode_text}?", default=True):
        print("\n已取消")
        return
    
    # 执行
    print("\n\n" + "=" * 50)
    print("  开始执行")
    print("=" * 50)
    
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    generator = DataGenerator()
    
    if config['mode'] == 'evaluate':
        # 仅评估模式
        adapter_path = input_text("Adapter 路径", default='./output/adapter', required=True)
        
        print("\n[1/1] 评估模型...")
        test_jsonl = Path(config['output_dir']) / "temp" / "test.jsonl"
        
        def get_jsonl_count(path):
            if not path.exists():
                return 0
            with open(path, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        
        test_count = get_jsonl_count(test_jsonl)
        
        if test_count > 0:
            print(f"  使用 {test_count} 条测试数据进行评估...")
        else:
            print("  ⚠ 未找到测试数据，从数据集随机抽样20%作为评估...")
            if config['dataset_dir']:
                train_qa, test_qa = generator.load_dataset_dir(
                    config['dataset_dir'], 
                    test_ratio=0.2
                )
                eval_jsonl = Path(config['output_dir']) / "temp" / "eval.jsonl"
                generator.save_to_jsonl(test_qa, str(eval_jsonl))
                test_jsonl = eval_jsonl
                print(f"  ✓ 随机抽取 {len(test_qa)} 条数据用于评估")
        
        finetune_config = FinetuneConfig(
            model_name=config['model_name'],
            model_path=adapter_path,
            data_path=str(test_jsonl),
            output_dir=config['output_dir'],
            device=config['device'],
        )
        finetuner = Finetuner(finetune_config)
        result = finetuner.evaluate(str(test_jsonl))
        print(f"  ✓ 评估完成")
        
        print("\n" + "=" * 50)
        print("  ✓ 评估完成!")
        print("=" * 50)
        return
    
    elif config['mode'] == 'convert':
        # 仅转换模式
        adapter_path = input_text("Adapter 路径", default='./output/adapter', required=True)
        
        print("\n[1/1] 转换 Ollama 模型...")
        converter = ModelConverter(
            base_model=config['model_path'] if config['model_path'] else config['model_name'],
            adapter_path=adapter_path,
            output_dir=config['output_dir'],
        )
        ollama_dir = converter.full_pipeline(config['ollama_name'])
        print(f"  ✓ Ollama 模型已创建: {ollama_dir}")
        
        print("\n" + "=" * 50)
        print("  ✓ 转换完成!")
        print("=" * 50)
        print(f"\n运行命令: ollama run {config['ollama_name']}")
        return
    
    # 完整训练流程
    train_jsonl = Path(config['output_dir']) / "temp" / "train.jsonl"
    test_jsonl = Path(config['output_dir']) / "temp" / "test.jsonl"
    train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    print("\n[1/4] 加载数据集...")
    
    if config['dataset_dir']:
        train_qa, test_qa = generator.load_dataset_dir(
            config['dataset_dir'], 
            test_ratio=config['test_ratio']
        )
    else:
        train_qa = generator.load_from_jsonl(config['train_data'])
        if config['test_data']:
            test_qa = generator.load_from_jsonl(config['test_data'])
        else:
            test_qa = []
            if config['test_ratio'] > 0 and train_qa:
                import random
                random.shuffle(train_qa)
                test_size = int(len(train_qa) * config['test_ratio'])
                test_qa = train_qa[:test_size]
                train_qa = train_qa[test_size:]
    
    generator.save_to_jsonl(train_qa, str(train_jsonl))
    generator.save_to_jsonl(test_qa, str(test_jsonl))
    print(f"  ✓ 训练集: {len(train_qa)} 条, 测试集: {len(test_qa)} 条")
    
    # 执行微调
    print("\n[2/4] 执行微调...")
    finetune_config = FinetuneConfig(
        model_name=config['model_name'],
        model_path=config['model_path'] or None,
        data_path=str(train_jsonl),
        output_dir=config['output_dir'],
        num_epochs=config['num_epochs'],
        device=config['device'],
    )
    
    finetuner = Finetuner(finetune_config)
    finetuner.prepare_data()
    finetuner.train()
    print(f"  ✓ 微调完成: {finetuner.get_adapter_path()}")
    
    # 评估模型
    print("\n[3/4] 评估模型...")
    
    def get_jsonl_count(path):
        if not path.exists():
            return 0
        with open(path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    
    test_count = get_jsonl_count(test_jsonl)
    
    if test_count > 0:
        print(f"  使用 {test_count} 条测试数据进行评估...")
        result = finetuner.evaluate(str(test_jsonl))
        print(f"  ✓ 评估完成")
    else:
        print("  ⚠ 测试集为空，从训练集随机抽样20%进行评估...")
        import random
        eval_size = max(1, int(len(train_qa) * 0.2))
        all_train = list(train_qa)
        random.shuffle(all_train)
        eval_data = all_train[:eval_size]
        eval_jsonl = Path(config['output_dir']) / "temp" / "eval.jsonl"
        generator.save_to_jsonl(eval_data, str(eval_jsonl))
        result = finetuner.evaluate(str(eval_jsonl))
        print(f"  ✓ 随机评估完成 (样本数: {len(eval_data)})")
    
    # 转换模型
    print("\n[4/4] 转换 Ollama 模型...")
    converter = ModelConverter(
        base_model=config['model_path'] if config['model_path'] else config['model_name'],
        adapter_path=finetuner.get_adapter_path(),
        output_dir=config['output_dir'],
    )
    ollama_dir = converter.full_pipeline(config['ollama_name'])
    print(f"  ✓ Ollama 模型已创建: {ollama_dir}")
    
    print("\n" + "=" * 50)
    print("  ✓ 全部完成!")
    print("=" * 50)
    print(f"\n运行命令: ollama run {config['ollama_name']}")
    
    if input_yes_no("\n是否清理临时文件?", default=False):
        import shutil
        if Path("output/temp").exists():
            shutil.rmtree("output/temp")
            print("  ✓ 已清理")


if __name__ == "__main__":
    main()
