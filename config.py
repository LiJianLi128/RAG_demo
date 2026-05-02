"""配置文件"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_ROOT = PROJECT_ROOT / "demo"
LOCAL_RERANK_MODEL_PATH = PROJECT_ROOT / "model" / "bge-reranker-base"

# OpenAI 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 嵌入模型配置
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 重排模型配置
RERANK_MODEL = str(LOCAL_RERANK_MODEL_PATH) if LOCAL_RERANK_MODEL_PATH.exists() else "BAAI/bge-reranker-base"
ENABLE_RERANK = True
RERANK_TOP_N = 6

# 文档分块配置
CHUNK_SIZE = 500  # 每块字符数
CHUNK_OVERLAP = 50  # 块之间重叠字符数

# 检索配置
TOP_K = 3  # 检索最相关的文档数量
SHOW_SOURCES = True
RETRIEVAL_MODE = "hybrid"
VECTOR_CANDIDATE_MULTIPLIER = 8
HYBRID_CANDIDATE_MULTIPLIER = 2

# 向量存储路径
VECTOR_STORE_PATH = str(PROJECT_ROOT / "vector_store")
INTERVIEW_ASSISTANT_STORE_PATH = str(DEMO_ROOT / "vector_store" / "interview_assistant")

# Lesson 6 LangChain demo 的 Chroma 持久化目录
# 不同数据文件落到不同子目录（按 stem 区分），避免 collection 串味
LESSON6_CHROMA_ROOT = str(PROJECT_ROOT / "vector_store" / "lesson6_chroma")

# 面试助手数据目录
INTERVIEW_ASSISTANT_DATA_ROOT = str(DEMO_ROOT / "data" / "interview_assistant")
INTERVIEW_ASSISTANT_CATEGORY_DIRS = {
    "highlights": str(Path(INTERVIEW_ASSISTANT_DATA_ROOT) / "highlights"),
    "interviews": str(Path(INTERVIEW_ASSISTANT_DATA_ROOT) / "interviews"),
    "knowledge": str(Path(INTERVIEW_ASSISTANT_DATA_ROOT) / "knowledge"),
}
