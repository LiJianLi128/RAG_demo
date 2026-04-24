"""示例：基本使用"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system import RAGSystem

def main():
    # 初始化 RAG 系统（不使用 OpenAI，只做检索）
    rag = RAGSystem(use_openai=False)

    # 添加示例文档
    documents = [
        "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。它通过从知识库中检索相关信息来增强语言模型的生成能力。",
        "向量数据库是 RAG 系统的核心组件。它将文本转换为向量，并支持高效的相似度搜索。常见的向量数据库包括 FAISS、Chroma 和 Pinecone。",
        "文本嵌入（Embedding）是将文本转换为数值向量的过程。这些向量能够捕捉文本的语义信息，使得相似的文本在向量空间中距离更近。",
        "在 RAG 系统中，文档分块（Chunking）是一个重要步骤。合理的分块策略可以提高检索质量和生成效果。",
        "Python 是实现 RAG 系统的常用语言。常用的库包括 LangChain、LlamaIndex、sentence-transformers 和 FAISS。"
    ]

    print("正在添加文档...")
    rag.add_documents(documents)

    # 保存向量存储
    rag.save()

    # 查询示例
    questions = [
        "什么是 RAG？",
        "向量数据库有哪些？",
        "如何实现 RAG 系统？"
    ]

    print("\n" + "="*50)
    print("开始查询")
    print("="*50)

    for question in questions:
        print(f"\n问题: {question}")
        print("-" * 50)
        answer = rag.query(question)
        print(f"答案:\n{answer}\n")

if __name__ == "__main__":
    main()
