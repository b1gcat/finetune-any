"""训练数据生成器 - 从标准/规定文档生成问答对"""
import json
import re
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path

@dataclass
class QA:
    """问答对"""
    instruction: str
    input: str
    output: str

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output
        }

class DataGenerator:
    """数据生成器"""

    def __init__(self):
        self.qa_pairs: List[QA] = []

    def generate_from_text(self, text: str, context: str = "") -> List[QA]:
        """从文本生成问答对"""
        qa_list = []

        sections = self._split_sections(text)
        for section in sections:
            qa_list.extend(self._extract_from_section(section, context))

        return qa_list

    def _split_sections(self, text: str) -> List[str]:
        """分割文本为章节"""
        patterns = [
            r'第[一二三四五六七八九十百\d]+[章节条款]',
            r'\d+\.\d+',
            r'第[一二三四五六七八九十百\d]+条',
            r'\([a-zA-Z]\)',
        ]

        pattern = '|'.join(patterns)
        parts = re.split(f'({pattern})', text)

        sections = []
        for i in range(1, len(parts), 2):
            header = parts[i]
            content_start = i + 1
            content = ''.join(parts[content_start:content_start+1] if content_start < len(parts) else [])
            sections.append(f"{header} {content}".strip())

        return [s for s in sections if len(s) > 20]

    def _extract_from_section(self, section: str, context: str) -> List[QA]:
        """从章节提取问答对"""
        qa_list = []

        if len(section) < 30:
            return qa_list

        keywords = {
            r'定义|概念|是指': ('definition', '什么是'),
            r'要求|应当|必须|需要': ('requirement', '有什么要求'),
            r'禁止|不得|不准': ('prohibition', '有什么禁止'),
            r'责任|义务|职责': ('duty', '有什么责任'),
            r'适用于|适用范围': ('scope', '适用范围'),
            r'程序|流程|步骤': ('procedure', '程序是什么'),
        }

        for pattern, (qtype, question_prefix) in keywords.items():
            if re.search(pattern, section):
                instruction = f"根据以下文档内容回答问题。回答时说明来源文档。"
                question = f"{context}关于{qtype}，{section[:100]}... 具体内容是什么？"

                qa_list.append(QA(
                    instruction=instruction,
                    input=question,
                    output=section
                ))

        if not qa_list:
            qa_list.append(QA(
                instruction="根据以下文档内容回答问题。回答时说明来源文档。",
                input=f"{context}这段内容讲了什么？",
                output=section[:500]
            ))

        return qa_list

    def generate_from_documents(self, documents: List[Dict]) -> List[QA]:
        """从文档列表生成问答对"""
        all_qa = []

        for doc in documents:
            filename = doc.get('filename', '')
            context = f"《{filename}》"

            full_text = "\n".join(p.get('text', '') for p in doc.get('pages', []))
            qa_list = self.generate_from_text(full_text, context)
            all_qa.extend(qa_list)

        return all_qa

    def save_to_jsonl(self, qa_list: List[QA], output_path: str):
        """保存为JSONL格式"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in qa_list:
                f.write(json.dumps(qa.to_dict(), ensure_ascii=False) + '\n')

        print(f"已生成 {len(qa_list)} 个问答对，保存到: {output_path}")
