"""配置文件"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 嵌入模型配置
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 文档分块配置
CHUNK_SIZE = 500  # 每块字符数
CHUNK_OVERLAP = 50  # 块之间重叠字符数

# 检索配置
TOP_K = 3  # 检索最相关的文档数量

# 向量存储路径
VECTOR_STORE_PATH = "./vector_store"
