"""文档处理模块：负责文档加载和分块"""
from typing import List

def load_text_file(file_path: str) -> str:
    """加载文本文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    将文本分块

    Args:
        text: 原始文本
        chunk_size: 每块大小
        overlap: 块之间重叠大小

    Returns:
        文本块列表
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        # 避免在单词中间切断（简单实现）
        if end < text_length and not text[end].isspace():
            last_space = chunk.rfind(' ')
            if last_space > 0:
                end = start + last_space
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if c]  # 过滤空块
