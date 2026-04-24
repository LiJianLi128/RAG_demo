"""向量存储模块：负责文档嵌入和检索"""
import os
import pickle
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class VectorStore:
    def __init__(self, model_name: str, store_path: str = "./vector_store"):
        """
        初始化向量存储

        Args:
            model_name: 嵌入模型名称
            store_path: 向量存储路径
        """
        self.model = SentenceTransformer(model_name)
        self.store_path = store_path
        self.index = None
        self.documents = []

        # 创建存储目录
        os.makedirs(store_path, exist_ok=True)

    def add_documents(self, documents: List[str]):
        """添加文档到向量存储"""
        print(f"正在生成 {len(documents)} 个文档的嵌入向量...")
        embeddings = self.model.encode(documents, show_progress_bar=True)

        # 创建或更新 FAISS 索引
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        print(f"已添加 {len(documents)} 个文档")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        搜索最相关的文档

        Args:
            query: 查询文本
            top_k: 返回最相关的 k 个文档

        Returns:
            (文档, 相似度分数) 的列表
        """
        if self.index is None or len(self.documents) == 0:
            return []

        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(distance)))

        return results

    def save(self):
        """保存向量存储到磁盘"""
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(self.store_path, "index.faiss"))
            with open(os.path.join(self.store_path, "documents.pkl"), 'wb') as f:
                pickle.dump(self.documents, f)
            print(f"向量存储已保存到 {self.store_path}")

    def load(self):
        """从磁盘加载向量存储"""
        index_path = os.path.join(self.store_path, "index.faiss")
        docs_path = os.path.join(self.store_path, "documents.pkl")

        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"已加载 {len(self.documents)} 个文档")
            return True
        return False
