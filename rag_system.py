"""RAG 核心模块：整合检索和生成"""
from typing import List, Optional
from openai import OpenAI
from vector_store import VectorStore
from document_processor import chunk_text, load_text_file
import config

class RAGSystem:
    def __init__(self, use_openai: bool = True):
        """
        初始化 RAG 系统

        Args:
            use_openai: 是否使用 OpenAI API（False 则只返回检索结果）
        """
        self.vector_store = VectorStore(
            model_name=config.EMBEDDING_MODEL,
            store_path=config.VECTOR_STORE_PATH
        )
        self.use_openai = use_openai

        if use_openai and config.OPENAI_API_KEY:
            self.client = OpenAI(
                api_key=config.OPENAI_API_KEY,
                base_url=config.OPENAI_BASE_URL
            )
        else:
            self.client = None
            if use_openai:
                print("警告：未配置 OpenAI API，将只返回检索结果")

    def add_documents_from_file(self, file_path: str):
        """从文件添加文档"""
        print(f"正在处理文件: {file_path}")
        text = load_text_file(file_path)
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        print(f"文档已分成 {len(chunks)} 块")
        self.vector_store.add_documents(chunks)

    def add_documents(self, documents: List[str]):
        """直接添加文档列表"""
        self.vector_store.add_documents(documents)

    def query(self, question: str, top_k: Optional[int] = None) -> str:
        """
        查询 RAG 系统

        Args:
            question: 用户问题
            top_k: 检索文档数量

        Returns:
            生成的答案
        """
        if top_k is None:
            top_k = config.TOP_K

        # 检索相关文档
        results = self.vector_store.search(question, top_k)

        if not results:
            return "抱歉，没有找到相关文档。"

        # 构建上下文
        context = "\n\n".join([doc for doc, _ in results])

        print(f"\n检索到 {len(results)} 个相关文档片段")

        # 如果没有配置 LLM，只返回检索结果
        if not self.client:
            return f"检索到的相关内容：\n\n{context}"

        # 使用 LLM 生成答案
        prompt = f"""基于以下上下文回答问题。如果上下文中没有相关信息，请说明无法回答。

上下文：
{context}

问题：{question}

回答："""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个helpful的助手，基于提供的上下文回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            return f"检索到的相关内容：\n\n{context}"

    def save(self):
        """保存向量存储"""
        self.vector_store.save()

    def load(self):
        """加载向量存储"""
        return self.vector_store.load()
