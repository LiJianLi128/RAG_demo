"""
演示跨语言检索的实际应用
"""
from sentence_transformers import SentenceTransformer
import numpy as np

print("加载多语言模型...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("\n" + "="*60)
print("场景：跨语言文档检索")
print("="*60)

# 文档库（混合中英文）
documents = [
    "Python是一种编程语言",
    "Machine learning is a subset of AI",
    "深度学习需要大量数据",
    "Natural language processing handles text",
    "RAG系统结合了检索和生成",
]

# 用户查询（中文）
query = "什么是机器学习？"

print(f"\n用户查询（中文）: {query}")
print("\n文档库:")
for i, doc in enumerate(documents):
    print(f"  {i+1}. {doc}")

# 编码
query_vec = model.encode([query])[0]
doc_vecs = model.encode(documents)

# 计算相似度
print("\n相似度排名:")
similarities = []
for i, doc_vec in enumerate(doc_vecs):
    sim = cosine_similarity(query_vec, doc_vec)
    similarities.append((i, sim, documents[i]))

similarities.sort(key=lambda x: x[1], reverse=True)

for rank, (idx, sim, doc) in enumerate(similarities, 1):
    lang = "中文" if any('\u4e00' <= c <= '\u9fff' for c in doc) else "英文"
    print(f"  {rank}. [{lang}] {doc}")
    print(f"     相似度: {sim:.4f}")

print("\n" + "="*60)
print("观察：")
print("="*60)
print("1. 中文查询能找到英文文档！")
print("2. 'Machine learning' 被识别为最相关（虽然是英文）")
print("3. 这就是跨语言RAG的基础")

# 反向测试：英文查询中文文档
print("\n" + "="*60)
print("反向测试：英文查询")
print("="*60)

query_en = "What is deep learning?"
print(f"\n用户查询（英文）: {query_en}")

query_vec_en = model.encode([query_en])[0]
similarities_en = []
for i, doc_vec in enumerate(doc_vecs):
    sim = cosine_similarity(query_vec_en, doc_vec)
    similarities_en.append((i, sim, documents[i]))

similarities_en.sort(key=lambda x: x[1], reverse=True)

print("\n相似度排名:")
for rank, (idx, sim, doc) in enumerate(similarities_en, 1):
    lang = "中文" if any('\u4e00' <= c <= '\u9fff' for c in doc) else "英文"
    print(f"  {rank}. [{lang}] {doc}")
    print(f"     相似度: {sim:.4f}")

print("\n结论：英文查询也能找到中文文档！")
