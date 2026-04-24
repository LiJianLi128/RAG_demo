"""示例：从文件加载文档"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system import RAGSystem

def main():
    sample_file = PROJECT_ROOT / "data" / "sample.txt"

    # 初始化 RAG 系统
    rag = RAGSystem(use_openai=False)

    # 尝试加载已有的向量存储
    if rag.load():
        print("已加载现有向量存储")
    else:
        print("未找到现有向量存储，将创建新的")

    # 从文件添加文档（需要先创建示例文件）
    try:
        rag.add_documents_from_file(str(sample_file))
        rag.save()
    except FileNotFoundError:
        print(f"未找到 {sample_file} 文件")
        print("请创建该文件或使用 examples/example_basic.py")
        return

    # 交互式查询
    print("\n" + "="*50)
    print("RAG 系统已就绪，输入问题进行查询（输入 'quit' 退出）")
    print("="*50 + "\n")

    while True:
        question = input("问题: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break

        if not question:
            continue

        print("-" * 50)
        answer = rag.query(question)
        print(f"答案:\n{answer}\n")

if __name__ == "__main__":
    main()
