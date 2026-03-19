"""文档解析器测试"""
import pytest
import tempfile
from pathlib import Path
from src.doc_parser.parser import DocumentParser, Document, Page

def test_document_full_text():
    doc = Document(
        filename="test.pdf",
        filepath="/tmp/test.pdf",
        pages=[Page(1, "第一页"), Page(2, "第二页")]
    )
    assert doc.full_text() == "第一页\n\n第二页"

def test_document_to_dict():
    doc = Document(
        filename="test.pdf",
        filepath="/tmp/test.pdf",
        pages=[Page(1, "测试内容")],
        metadata={"title": "测试文档"}
    )
    data = doc.to_dict()
    assert data["filename"] == "test.pdf"
    assert data["metadata"]["title"] == "测试文档"

def test_supported_formats():
    parser = DocumentParser("/tmp")
    assert ".pdf" in parser.SUPPORTED_FORMATS
    assert ".docx" in parser.SUPPORTED_FORMATS
    assert ".txt" in parser.SUPPORTED_FORMATS
    assert ".md" in parser.SUPPORTED_FORMATS
