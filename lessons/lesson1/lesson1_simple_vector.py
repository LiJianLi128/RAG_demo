"""
第一课：最简单的向量检索
目标：理解文本如何变成向量，以及如何计算相似度
"""
import numpy as np

# 我们的文档库
documents = [
    "我喜欢吃苹果",
    "我爱吃水果",
    "今天天气很好"
]

# 用户的查询
query = "我想吃苹果"

print("=" * 60)
print("第一课：向量检索的本质")
print("=" * 60)

# 步骤1：构建词汇表（所有出现过的词）
def build_vocabulary(texts):
    """从所有文本中提取不重复的词"""
    vocab = set()
    for text in texts:
        # 简单分词（实际应该用 jieba 等工具）
        words = list(text)  # 这里简化为字符级别
        vocab.update(words)
    return sorted(list(vocab))

all_texts = documents + [query]
vocabulary = build_vocabulary(all_texts)

print(f"\n步骤1：构建词汇表")
print(f"词汇表: {vocabulary}")
print(f"词汇表大小: {len(vocabulary)}")

# 步骤2：将文本转换为向量
def text_to_vector(text, vocabulary):
    """
    将文本转换为向量
    向量的每个维度代表一个词，值是该词出现的次数
    """
    vector = np.zeros(len(vocabulary))
    for char in text:
        if char in vocabulary:
            idx = vocabulary.index(char)
            vector[idx] += 1
    return vector

print(f"\n步骤2：文本向量化")
doc_vectors = []
for i, doc in enumerate(documents):
    vec = text_to_vector(doc, vocabulary)
    doc_vectors.append(vec)
    print(f"文档{i+1} '{doc}'")
    print(f"  向量: {vec}")
    print(f"  非零维度: {np.count_nonzero(vec)}")

query_vector = text_to_vector(query, vocabulary)
print(f"\n查询 '{query}'")
print(f"  向量: {query_vector}")

# 步骤3：计算相似度（余弦相似度）
def cosine_similarity(vec1, vec2):
    """
    余弦相似度 = 两个向量的点积 / (向量长度的乘积)
    值域：[-1, 1]，越接近1越相似
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot_product / (norm1 * norm2)

print(f"\n步骤3：计算相似度")
print("-" * 60)
similarities = []
for i, doc_vec in enumerate(doc_vectors):
    sim = cosine_similarity(query_vector, doc_vec)
    similarities.append((i, sim))
    print(f"查询 vs 文档{i+1}: {sim:.4f}")

# 步骤4：找到最相似的文档
similarities.sort(key=lambda x: x[1], reverse=True)
best_match_idx = similarities[0][0]
best_match_score = similarities[0][1]

print(f"\n步骤4：检索结果")
print("=" * 60)
print(f"最相关的文档: 文档{best_match_idx+1}")
print(f"内容: '{documents[best_match_idx]}'")
print(f"相似度得分: {best_match_score:.4f}")
print("=" * 60)

# 思考题
print("\n💡 思考题：")
print("1. 为什么 '我喜欢吃苹果' 和 '我想吃苹果' 相似度高？")
print("2. 这种方法有什么问题？（提示：'喜欢'和'爱'是近义词）")
print("3. 如果文档有10万个，这种方法效率如何？")
