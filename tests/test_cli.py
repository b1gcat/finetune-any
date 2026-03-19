"""CLI测试"""
import pytest
from click.testing import CliRunner
from main import cli

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert '文档微调程序' in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])
    assert result.exit_code == 0
    assert '0.1.0' in result.output


def test_parse_command_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['parse', '--help'])
    assert result.exit_code == 0
    assert '解析文档' in result.output


def test_generate_command_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['generate', '--help'])
    assert result.exit_code == 0
    assert '生成训练数据' in result.output


def test_finetune_command_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['finetune', '--help'])
    assert result.exit_code == 0
    assert '执行微调训练' in result.output


def test_convert_command_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['convert', '--help'])
    assert result.exit_code == 0
    assert '转换模型为Ollama格式' in result.output


def test_all_command_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['all', '--help'])
    assert result.exit_code == 0
    assert '全流程' in result.output
