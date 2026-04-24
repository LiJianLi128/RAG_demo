"""第六课：使用 LangChain 搭建本地检索版 RAG demo"""
from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_documents(file_path: str) -> list[Document]:
    text = Path(file_path).read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": file_path})]


def build_vector_store(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[FAISS, list[Document]]:
    documents = load_documents(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, chunks


def print_chunks(chunks: list[Document]):
    print("\n" + "=" * 60)
    print(f"文档已切分为 {len(chunks)} 个 chunk")
    print("=" * 60)

    for chunk in chunks:
        preview = chunk.page_content[:120].replace("\n", " ")
        print(f"chunk #{chunk.metadata['chunk_id']}: {preview}")
        print("-" * 60)


def search_demo(vector_store: FAISS, question: str, top_k: int):
    print(f"\n问题: {question}")
    print("-" * 60)

    results = vector_store.similarity_search_with_score(question, k=top_k)

    for rank, (doc, score) in enumerate(results, start=1):
        print(f"Top {rank} | chunk #{doc.metadata.get('chunk_id')} | L2 distance: {score:.4f}")
        print(doc.page_content[:200].strip())
        print("-" * 60)


def parse_args():
    parser = ArgumentParser(description="LangChain 本地检索参数实验")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument(
        "--file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "我司git的操作规范文档.md"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    file_path = Path(args.file)

    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    if not file_path.exists():
        print(f"未找到文件: {file_path}")
        return

    print("正在构建 LangChain 本地检索 demo...")
    print(f"数据文件: {file_path}")
    print(f"embedding 模型: {config.EMBEDDING_MODEL}")
    print(
        f"chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}, top_k={args.top_k}"
    )

    vector_store, chunks = build_vector_store(
        file_path=file_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print_chunks(chunks)

    questions = [
        "git add . 为什么不推荐？",
    ]

    print("\n" + "=" * 60)
    print("开始检索演示")
    print("=" * 60)
    for question in questions:
        search_demo(vector_store, question, args.top_k)

    print("\n建议你直接做 3 组对比实验：")
    print(
        "1. python lessons/lesson6/lesson6_langchain_retrieval.py --chunk-size 200 --chunk-overlap 20 --top-k 2"
    )
    print(
        "2. python lessons/lesson6/lesson6_langchain_retrieval.py --chunk-size 500 --chunk-overlap 50 --top-k 3"
    )
    print(
        "3. python lessons/lesson6/lesson6_langchain_retrieval.py --chunk-size 900 --chunk-overlap 100 --top-k 4"
    )
    print("\n如需切换数据源，可额外传入 --file data/sample.txt")


if __name__ == "__main__":
    main()
