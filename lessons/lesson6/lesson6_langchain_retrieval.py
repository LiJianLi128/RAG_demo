"""第六课：使用 LangChain 搭建本地检索版 RAG demo"""
from argparse import ArgumentParser
from pathlib import Path
import pickle
import re
import shutil
import sys

import jieba
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

import config


def _slugify_for_collection(name: str) -> str:
    # Chroma collection name 限制：3-63 字符、[a-zA-Z0-9._-]、不能以特殊字符开头/结尾。
    # 把不合规的字符（包括中文）统一替换成下划线，避免直接传中文文件名导致建库失败。
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    slug = re.sub(r"^[._-]+|[._-]+$", "", slug)
    if len(slug) < 3:
        slug = f"col_{slug}"
    return slug[:63] or "default_collection"


def load_documents(file_path: str) -> list[Document]:
    text = Path(file_path).read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": file_path})]


def load_reranker(model_name: str) -> CrossEncoder:
    # reranker 只在启用时加载，避免默认实验也去下载额外模型。
    return CrossEncoder(model_name)


def split_sentences(text: str) -> list[str]:
    # 先按句号/问号/感叹号等边界切句，给后续“按语义合并句子”做准备。
    parts = re.split(r"(?<=[。！？；.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def split_markdown_sections(text: str) -> list[str]:
    # 先按 Markdown 小节切开，避免把相距很远但恰好语义相近的内容误拼到同一个 chunk。
    heading_pattern = re.compile(r"(?m)^(#{2,3}\s+.+)$")
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return [text.strip()] if text.strip() else []

    sections: list[str] = []
    if matches[0].start() > 0:
        prefix = text[: matches[0].start()].strip()
        if prefix:
            sections.append(prefix)

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    return sections


def tokenize_for_bm25(text: str) -> list[str]:
    # BM25 对中文不能直接按空格切词，这里先用 jieba 分词，再交给 BM25 建索引。
    normalized_text = re.sub(r"\s+", " ", text)
    tokens = [token.strip() for token in jieba.lcut(normalized_text) if token.strip()]
    return tokens or [normalized_text]


def semantic_chunk_documents(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    target_chunk_size: int,
    similarity_threshold: float,
) -> list[Document]:
    # 核心思路：先按 section 切，再按句子向量的相似度决定哪些句子应放在同一个 chunk 里。
    semantic_chunks: list[Document] = []

    for document in documents:
        for section in split_markdown_sections(document.page_content):
            section_metadata = document.metadata.copy()
            section_title = section.splitlines()[0].strip()
            section_metadata["section_title"] = section_title

            sentences = split_sentences(section)
            if not sentences:
                continue

            # 先给每个句子做 embedding，后面通过“相邻句是否仍然相近”来决定是否断块。
            sentence_vectors = embeddings.embed_documents(sentences)
            current_sentences = [sentences[0]]
            current_length = len(sentences[0])

            for index in range(1, len(sentences)):
                previous_vector = np.array(sentence_vectors[index - 1])
                current_vector = np.array(sentence_vectors[index])
                similarity = float(
                    np.dot(previous_vector, current_vector)
                    / (np.linalg.norm(previous_vector) * np.linalg.norm(current_vector))
                )

                next_sentence = sentences[index]
                next_length = len(next_sentence)
                next_text = "\n".join(current_sentences + [next_sentence])
                has_command_pattern = bool(re.search(r"`git [^`]+`|```bash", next_text))
                # 触发断块的三种情况：
                # 1. 相邻句语义已经明显变远；
                # 2. 当前 chunk 已接近长度上限；
                # 3. 当前内容进入命令/代码块，提前拆开更利于命令型查询命中。
                should_split = (
                    similarity < similarity_threshold
                    and current_length >= target_chunk_size // 2
                ) or current_length + next_length > target_chunk_size or (
                    has_command_pattern and current_length >= target_chunk_size // 3
                )

                if should_split:
                    semantic_chunks.append(
                        Document(
                            page_content="\n".join(current_sentences),
                            metadata=section_metadata.copy(),
                        )
                    )
                    current_sentences = [next_sentence]
                    current_length = next_length
                    continue

                current_sentences.append(next_sentence)
                current_length += next_length

            semantic_chunks.append(
                Document(
                    page_content="\n".join(current_sentences),
                    metadata=section_metadata.copy(),
                )
            )

    return semantic_chunks


def build_vector_store(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
    chunking_mode: str,
    similarity_threshold: float,
    top_k: int,
    rebuild: bool = False,
) -> tuple[Chroma, BM25Retriever, list[Document]]:
    # 同时构建两套检索能力：
    # - Chroma 负责语义相似度召回（持久化到磁盘，第二次起秒载）
    # - BM25 负责关键词/命令字面匹配（不持久化，从 chunks 毫秒级重建）
    #
    # 为什么用 Chroma 而不是 FAISS save_local：
    #   - Chroma 是真正的"向量数据库"（带 collection / metadata filter），不只是 index 文件
    #   - LangChain 抽象层一致，将来想换 Qdrant / Milvus / pgvector，业务代码几乎不用动
    #   - 落盘方案天然 (SQLite + parquet)，比手动 pickle index.faiss + documents.pkl 稳
    cache_name = _slugify_for_collection(Path(file_path).stem)
    chroma_dir = Path(config.LESSON6_CHROMA_ROOT) / cache_name
    chunks_path = chroma_dir / "chunks.pkl"

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    # 已有缓存且不强制重建：直接从持久化目录加载，跳过 embedding 计算
    if not rebuild and chunks_path.exists():
        print(f"[缓存] 从 {chroma_dir} 加载已有索引...")
        vector_store = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=embeddings,
            collection_name=cache_name,
        )
        with chunks_path.open("rb") as file:
            chunks = pickle.load(file)
        # BM25 故意不持久化：从 chunks 重建非常快（毫秒级词频统计），
        # 而 pickle BM25 对象会引入版本兼容风险（rank_bm25 内部状态变化时会崩）。
        bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=tokenize_for_bm25)
        bm25_retriever.k = max(1, min(len(chunks), top_k * 2))
        return vector_store, bm25_retriever, chunks

    # 强制重建时先清空目录，避免重复插入到同一个 collection 里
    if rebuild and chroma_dir.exists():
        print(f"[强制重建] 删除旧索引 {chroma_dir}...")
        shutil.rmtree(chroma_dir)

    print(f"[构建] 正在为 {file_path} 构建新索引到 {chroma_dir}...")
    documents = load_documents(file_path)

    if chunking_mode == "semantic":
        chunks = semantic_chunk_documents(
            documents=documents,
            embeddings=embeddings,
            target_chunk_size=chunk_size,
            similarity_threshold=similarity_threshold,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    chroma_dir.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(chroma_dir),
        collection_name=cache_name,
    )
    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        preprocess_func=tokenize_for_bm25,
    )
    bm25_retriever.k = max(1, min(len(chunks), top_k * 2))

    # 把 chunks 单独 pickle 一份，下次启动直接读这个就能重建 BM25 + 还原 metadata
    with chunks_path.open("wb") as file:
        pickle.dump(chunks, file)

    print(f"[完成] 索引已落盘（共 {len(chunks)} 个 chunk），下次启动会从缓存直接加载")
    return vector_store, bm25_retriever, chunks


def print_chunks(chunks: list[Document]):
    print("\n" + "=" * 60)
    print(f"文档已切分为 {len(chunks)} 个 chunk")
    print("=" * 60)

    for chunk in chunks:
        preview = chunk.page_content[:120].replace("\n", " ")
        print(f"chunk #{chunk.metadata['chunk_id']}: {preview}")
        print("-" * 60)


def reciprocal_rank_fusion(rankings: list[list[Document]], top_k: int) -> list[tuple[Document, float]]:
    # RRF 不直接比较不同检索器的原始分数，而是比较“各自排第几名”，更容易把两边都靠前的结果提上来。
    fused_scores: dict[int, float] = {}
    doc_lookup: dict[int, Document] = {}

    for ranking in rankings:
        for rank, document in enumerate(ranking, start=1):
            chunk_id = document.metadata["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1 / (60 + rank)
            doc_lookup[chunk_id] = document

    sorted_chunk_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)
    return [(doc_lookup[chunk_id], fused_scores[chunk_id]) for chunk_id in sorted_chunk_ids[:top_k]]


def rerank_results(
    question: str,
    results: list[tuple[Document, float]],
    reranker: CrossEncoder,
    top_k: int,
) -> list[tuple[Document, float]]:
    # 在少量候选上做 query-document 精排，让“召回到了但排位不稳”的结果有机会被提到前面。
    if not results:
        return []

    pairs = [(question, document.page_content) for document, _ in results]
    scores = reranker.predict(pairs)
    reranked = sorted(
        [(document, float(score)) for (document, _), score in zip(results, scores)],
        key=lambda item: item[1],
        reverse=True,
    )
    return reranked[:top_k]


def print_result_block(title: str, results: list[tuple[Document, float]], score_label: str):
    # 把每一路检索结果分别打出来，便于判断“是向量有偏差，还是 BM25 有偏差，还是融合策略有偏差”。
    print(f"\n[{title}]")
    for rank, (doc, score) in enumerate(results, start=1):
        section_title = doc.metadata.get("section_title", "")
        print(
            f"Top {rank} | chunk #{doc.metadata.get('chunk_id')} | {score_label}: {score:.4f}"
            + (f" | section: {section_title}" if section_title else "")
        )
        print(doc.page_content[:200].strip())
        print("-" * 60)


def hybrid_search(
    vector_store: Chroma,
    bm25_retriever: BM25Retriever,
    question: str,
    top_k: int,
    reranker: CrossEncoder | None = None,
    rerank_top_n: int | None = None,
) -> tuple[
    list[tuple[Document, float]],
    list[tuple[Document, float]],
    list[tuple[Document, float]],
    list[tuple[Document, float]],
]:
    # 先各自多取一些候选，再做融合；这样 hybrid 才有机会把“单路排第二但总体更合理”的结果提上来。
    vector_results = vector_store.similarity_search_with_score(question, k=top_k * 2)
    vector_docs = [doc for doc, _ in vector_results]
    bm25_docs = bm25_retriever.invoke(question)[: top_k * 2]
    bm25_results = [(doc, float(top_k * 2 - rank + 1)) for rank, doc in enumerate(bm25_docs, start=1)]
    fused_results = reciprocal_rank_fusion([vector_docs, bm25_docs], max(top_k, rerank_top_n or top_k))

    reranked_results: list[tuple[Document, float]] = []
    if reranker is not None:
        candidate_count = rerank_top_n or top_k * 2
        reranked_results = rerank_results(
            question,
            fused_results[:candidate_count],
            reranker,
            top_k,
        )

    return vector_results, bm25_results, fused_results, reranked_results


def search_demo(
    vector_store: Chroma,
    bm25_retriever: BM25Retriever,
    question: str,
    top_k: int,
    retrieval_mode: str,
    reranker: CrossEncoder | None = None,
    rerank_top_n: int | None = None,
):
    # 这里保留统一的检索入口，方便切换 pure vector / hybrid 两种模式做对比实验。
    print(f"\n问题: {question}")
    print("-" * 60)

    if retrieval_mode == "hybrid":
        vector_results, bm25_results, fused_results, reranked_results = hybrid_search(
            vector_store,
            bm25_retriever,
            question,
            top_k,
            reranker,
            rerank_top_n,
        )
        print_result_block("vector", vector_results[:top_k], "L2 distance")
        print_result_block("bm25", bm25_results[:top_k], "Rank score")
        print_result_block("hybrid", fused_results, "RRF score")
        if reranked_results:
            print_result_block("rerank", reranked_results, "CrossEncoder score")
        return

    results = vector_store.similarity_search_with_score(question, k=top_k)
    print_result_block("vector", results, "L2 distance")


def run_query_variants(
    vector_store: Chroma,
    bm25_retriever: BM25Retriever,
    top_k: int,
    retrieval_mode: str,
    reranker: CrossEncoder | None = None,
    rerank_top_n: int | None = None,
):
    # 用多种近义问法重复测试，观察检索问题到底来自 chunk、召回，还是 query 表述本身。
    query_variants = [
        "git add . 为什么不推荐？",
        "为什么不推荐 git add .",
        "git add . 会带来什么问题",
        "提交前为什么不要使用 git add .",
    ]

    for question in query_variants:
        search_demo(
            vector_store,
            bm25_retriever,
            question,
            top_k,
            retrieval_mode,
            reranker,
            rerank_top_n,
        )


def parse_args():
    parser = ArgumentParser(description="LangChain 本地检索参数实验")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument(
        "--retrieval-mode",
        choices=["vector", "hybrid"],
        default="hybrid",
    )
    parser.add_argument(
        "--chunking-mode",
        choices=["recursive", "semantic"],
        default="semantic",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.72,
    )
    parser.add_argument(
        "--enable-rerank",
        action="store_true",
    )
    parser.add_argument(
        "--rerank-model",
        type=str,
        default=config.RERANK_MODEL,
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "我司git的操作规范文档.md"),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="强制删除已有的 Chroma 索引并重新构建（默认有缓存就直接 load）",
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
    if args.enable_rerank:
        print(f"rerank 模型: {args.rerank_model}")
    print(
        f"retrieval_mode={args.retrieval_mode}, chunking_mode={args.chunking_mode}, chunk_size={args.chunk_size}, "
        f"chunk_overlap={args.chunk_overlap}, similarity_threshold={args.similarity_threshold}, top_k={args.top_k}"
    )

    vector_store, bm25_retriever, chunks = build_vector_store(
        file_path=file_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_mode=args.chunking_mode,
        similarity_threshold=args.similarity_threshold,
        top_k=args.top_k,
        rebuild=args.rebuild,
    )
    reranker = load_reranker(args.rerank_model) if args.enable_rerank else None
    print_chunks(chunks)

    print("\n" + "=" * 60)
    print("开始检索演示")
    print("=" * 60)
    run_query_variants(
        vector_store,
        bm25_retriever,
        args.top_k,
        args.retrieval_mode,
        reranker,
        args.rerank_top_n,
    )

    print("\n建议你直接做 3 组对比实验：")
    print(
        "1. python lessons/lesson6/lesson6_langchain_retrieval.py --retrieval-mode hybrid --chunking-mode semantic --chunk-size 500 --top-k 3"
    )
    print(
        "2. python lessons/lesson6/lesson6_langchain_retrieval.py --retrieval-mode vector --chunking-mode semantic --chunk-size 500 --top-k 3"
    )
    print(
        "3. python lessons/lesson6/lesson6_langchain_retrieval.py --retrieval-mode hybrid --chunking-mode recursive --chunk-size 500 --chunk-overlap 50 --top-k 3"
    )
    print("\n如需切换数据源，可额外传入 --file data/sample.txt")


if __name__ == "__main__":
    main()
