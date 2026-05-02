"""第六课收尾：在 LangChain 检索的基础上接入 LLM 生成。

分 3 个阶段演示同一件事，从"朴素拼字符串"到"完整 LCEL"，便于体感对比。
"""
from argparse import ArgumentParser
import os
import sys
from pathlib import Path

LESSON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LESSON_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(LESSON_DIR) not in sys.path:
    sys.path.insert(0, str(LESSON_DIR))

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from openai import APIStatusError

import config
from lesson6_langchain_retrieval import build_vector_store, hybrid_search, load_reranker


def _dump_api_error(e: APIStatusError) -> None:
    # 网关 4xx/5xx 时，SDK 默认只抛短消息。把原始 body 和 status 打出来，方便定位是
    # 内容过滤、长度超限、额度不足，还是别的策略。
    print(f"[网关返回] HTTP {e.status_code}")
    try:
        print(f"[body] {e.response.text[:1000]}")
    except Exception:
        print(f"[body] {e!r}")


def format_docs_for_prompt(results: list[tuple[Document, float]]) -> str:
    # 把 (doc, score) 列表转成给 LLM 看的上下文字符串。
    # 保留 chunk_id 是为了让 LLM 在回答里能引用，方便我们后续做"溯源"实验。
    blocks = []
    for index, (doc, _) in enumerate(results, start=1):
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        blocks.append(f"[来源{index} | chunk #{chunk_id}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_llm(model_name: str | None = None) -> ChatOpenAI:
    # 复用 config 里的 OPENAI_API_KEY 和 OPENAI_BASE_URL：
    # GPT-OSS、GLM、OpenAI、DeepSeek 都填同一对环境变量，业务代码不用动，这正是 OpenAI-compatible 的红利。
    # model_name 显式传入用于多模型对比；不传时回退到 .env 里的 LLM_MODEL。
    api_key = config.OPENAI_API_KEY
    base_url = config.OPENAI_BASE_URL
    if not api_key:
        raise SystemExit("未读到 OPENAI_API_KEY。请先 cp .env.example .env 并填入你的 key。")

    resolved_model = model_name or os.getenv("LLM_MODEL", "gpt-oss-120b")
    # 第三方 OpenAI-compatible 网关（如 rosmontis）常用 WAF 拦截官方 SDK 默认带的
    # User-Agent: OpenAI/Python 和 x-stainless-* 遥测 header。这里覆盖成中性值绕开。
    safe_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "x-stainless-arch": "",
        "x-stainless-lang": "",
        "x-stainless-os": "",
        "x-stainless-package-version": "",
        "x-stainless-runtime": "",
        "x-stainless-runtime-version": "",
    }
    return ChatOpenAI(
        model=resolved_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.3,  # RAG 场景下偏低一点，减少胡编
        default_headers=safe_headers,
    )


def stage_1_naive(question: str, results: list[tuple[Document, float]], llm: ChatOpenAI) -> str:
    # 阶段 1：完全手写。等价于你 rag_system.py:135 现在的写法，只是 client 换成了 LangChain 的 ChatOpenAI。
    context = format_docs_for_prompt(results)
    prompt_text = f"""基于以下上下文回答问题。如果上下文里没有相关信息，请直接说"无法回答"，不要编造。

上下文：
{context}

问题：{question}

回答："""
    response = llm.invoke(prompt_text)
    return response.content


def stage_2_prompt_template(
    question: str,
    results: list[tuple[Document, float]],
    llm: ChatOpenAI,
) -> str:
    # 阶段 2：用 ChatPromptTemplate 把 system / user 两个角色显式分开。
    # 注意 prompt | llm 已经是一个最小的 LCEL 流水线了。
    context = format_docs_for_prompt(results)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一个 RAG 助手。只基于提供的上下文回答问题。"
            "如果上下文里没有相关信息，请直接说'无法回答'，不要自行编造。"
            "回答时尽量引用具体的来源编号（如[来源1]）。",
        ),
        ("user", "上下文：\n{context}\n\n问题：{question}"),
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content


def stage_3_lcel_chain(
    args,
    vector_store,
    bm25_retriever,
    reranker,
    llm: ChatOpenAI,
) -> str:
    # 阶段 3：完整 LCEL。把"检索"也包成 Runnable，外部只需要 chain.invoke(question)。
    # 这才是 LangChain 想推广的写法：retriever、prompt、llm、parser 全部是 Runnable，用 | 串起来。
    def retrieve(question: str):
        _, _, _, reranked = hybrid_search(
            vector_store,
            bm25_retriever,
            question,
            args.top_k,
            reranker,
            args.rerank_top_n,
        )
        return reranked

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一个 RAG 助手。只基于提供的上下文回答问题。"
            "如果上下文里没有相关信息，请直接说'无法回答'，不要自行编造。"
            "回答时尽量引用具体的来源编号（如[来源1]）。",
        ),
        ("user", "上下文：\n{context}\n\n问题：{question}"),
    ])

    # 这里的关键点：
    # - context 这一支：先 retrieve(question)，再 format_docs_for_prompt，得到字符串
    # - question 这一支：直接透传原问题
    # 两支都准备好之后塞进 prompt 的占位符。
    chain = (
        {
            "context": RunnableLambda(retrieve) | RunnableLambda(format_docs_for_prompt),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(args.question)


def print_retrieval(results: list[tuple[Document, float]]):
    print("\n[检索结果（rerank 后）]")
    for rank, (doc, score) in enumerate(results, start=1):
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  Top {rank} | rerank score: {score:.4f} | chunk #{doc.metadata.get('chunk_id')}")
        print(f"    {preview}...")


def stage_compare(
    args,
    vector_store,
    bm25_retriever,
    reranker,
    model_names: list[str],
):
    # 多模型对比：同一份检索结果，依次喂给多个模型，让差异肉眼可见。
    # 这是 RAG 工程师"选型直觉"的训练方法 —— 不靠 benchmark 排名，看谁在你的检索结果上回答得最稳。
    _, _, _, reranked = hybrid_search(
        vector_store,
        bm25_retriever,
        args.question,
        args.top_k,
        reranker,
        args.rerank_top_n,
    )
    if not reranked:
        print("没有检索到任何结果，无法生成回答。")
        return
    print_retrieval(reranked)

    print("\n" + "=" * 60)
    print(f"开始多模型对比：共 {len(model_names)} 个模型")
    print("=" * 60)

    for index, model_name in enumerate(model_names, start=1):
        print(f"\n[{index}/{len(model_names)}] 模型: {model_name}")
        print("-" * 60)
        try:
            llm = build_llm(model_name=model_name)
            answer = stage_2_prompt_template(args.question, reranked, llm)
            print(answer)
        except Exception as e:
            # 一个模型挂了不影响其他模型继续跑
            print(f"[失败] {type(e).__name__}: {e}")


def parse_args():
    parser = ArgumentParser(description="第六课收尾：检索 + 生成（OpenAI-compatible 接口）")
    parser.add_argument(
        "--stage",
        choices=["1", "2", "3"],
        default="3",
        help="1=朴素 f-string  2=PromptTemplate  3=完整 LCEL Chain",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="为什么不推荐使用 git add . 来提交代码？",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=0.72)
    parser.add_argument("--rerank-top-n", type=int, default=6)
    parser.add_argument(
        "--file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "我司git的操作规范文档.md"),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="多模型对比模式：同一问题依次跑多个模型，对比回答差异",
    )
    parser.add_argument(
        "--compare-models",
        type=str,
        default="gpt-oss-120b,glm-5-turbo,glm-5.1,kimi-k2.5,qwen3.5-397b-a17b",
        help="对比模式下要跑的模型列表，逗号分隔",
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

    print("=" * 60)
    print(f"模式: {'多模型对比' if args.compare else f'stage {args.stage}'}")
    print(f"问题: {args.question}")
    if not args.compare:
        print(f"模型: {os.getenv('LLM_MODEL', 'gpt-oss-120b')}")
    print(f"接口: {config.OPENAI_BASE_URL}")
    print("=" * 60)

    # 检索基建：所有模式共用
    print("\n正在构建检索基建（vector + BM25 + reranker）...")
    vector_store, bm25_retriever, _chunks = build_vector_store(
        file_path=str(file_path),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_mode="semantic",
        similarity_threshold=args.similarity_threshold,
        top_k=args.top_k,
        rebuild=args.rebuild,
    )
    reranker = load_reranker(config.RERANK_MODEL)

    # 多模型对比模式：单独走 stage_compare
    if args.compare:
        model_names = [m.strip() for m in args.compare_models.split(",") if m.strip()]
        stage_compare(args, vector_store, bm25_retriever, reranker, model_names)
        return

    llm = build_llm()

    if args.stage in ("1", "2"):
        # stage 1/2：手动检索一次，再交给 stage 函数
        _, _, _, reranked = hybrid_search(
            vector_store,
            bm25_retriever,
            args.question,
            args.top_k,
            reranker,
            args.rerank_top_n,
        )
        if not reranked:
            print("没有检索到任何结果，无法生成回答。")
            return
        print_retrieval(reranked)

        print("\n[LLM 生成]")
        try:
            if args.stage == "1":
                answer = stage_1_naive(args.question, reranked, llm)
            else:
                answer = stage_2_prompt_template(args.question, reranked, llm)
        except APIStatusError as e:
            _dump_api_error(e)
            return
    else:
        # stage 3：检索包进 chain，外部只 invoke 一次
        print("\n[LLM 生成（LCEL chain，自带检索）]")
        answer = stage_3_lcel_chain(args, vector_store, bm25_retriever, reranker, llm)

    print(answer)

    print("\n" + "=" * 60)
    print("建议你按这个顺序跑对比实验：")
    rel = Path(__file__).relative_to(PROJECT_ROOT).as_posix()
    print("\n  ① 先跑 3 个 stage，观察代码抽象程度的演进：")
    print(f"     python {rel} --stage 1")
    print(f"     python {rel} --stage 2")
    print(f"     python {rel} --stage 3")
    print("\n  ② 再跑多模型对比，观察同一上下文不同模型回答差异（这一节最值得做的实验）：")
    print(f"     python {rel} --compare")
    print("\n  ③ 换问题再跑：")
    print(f"     python {rel} --compare --question '提交前怎么避免误暂存？'")


if __name__ == "__main__":
    main()
