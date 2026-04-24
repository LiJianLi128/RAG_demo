"""
第二课：词嵌入的魔力
使用真实的嵌入模型，看看语义相似度
"""
from sentence_transformers import SentenceTransformer
import numpy as np

print("正在加载嵌入模型（首次运行会下载，约400MB）...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("模型加载完成！\n")

def get_similarity(text1, text2):
    """计算两个文本的语义相似度"""
    embeddings = model.encode([text1, text2])
    # 计算余弦相似度
    cos_sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return cos_sim

print("="*60)
print("词嵌入 vs 词袋模型：对比实验")
print("="*60)

# 实验1：近义词
print("\n实验1：近义词识别")
print("-"*60)
pairs1 = [
    ("我喜欢吃苹果", "我爱吃苹果"),
    ("我喜欢吃苹果", "我讨厌吃苹果"),
]

for text1, text2 in pairs1:
    sim = get_similarity(text1, text2)
    print(f"'{text1}' vs '{text2}'")
    print(f"  语义相似度: {sim:.4f}")

print("\n观察：'喜欢'和'爱'的相似度应该比'喜欢'和'讨厌'高")

# 实验2：同类词
print("\n实验2：同类词识别")
print("-"*60)
pairs2 = [
    ("我想吃苹果", "我想吃香蕉"),
    ("我想吃苹果", "我想吃汽车"),
]

for text1, text2 in pairs2:
    sim = get_similarity(text1, text2)
    print(f"'{text1}' vs '{text2}'")
    print(f"  语义相似度: {sim:.4f}")

print("\n观察：'苹果'和'香蕉'的相似度应该比'苹果'和'汽车'高")

# 实验3：语序敏感性
print("\n实验3：语序理解")
print("-"*60)
pairs3 = [
    ("猫咬了狗", "狗咬了猫"),
    ("猫咬了狗", "猫被狗咬了"),
]

for text1, text2 in pairs3:
    sim = get_similarity(text1, text2)
    print(f"'{text1}' vs '{text2}'")
    print(f"  语义相似度: {sim:.4f}")

print("\n观察：词嵌入能理解语序和语义的差异")

# 实验4：跨语言理解（这个模型支持多语言）
print("\n实验4：跨语言语义")
print("-"*60)
pairs4 = [
    ("我喜欢吃苹果", "I like eating apples"),
    ("今天天气很好", "I like eating apples"),
]

for text1, text2 in pairs4:
    sim = get_similarity(text1, text2)
    print(f"'{text1}' vs '{text2}'")
    print(f"  语义相似度: {sim:.4f}")

print("\n观察：相同意思的中英文句子相似度高！")

# 查看向量维度
print("\n" + "="*60)
print("向量维度对比")
print("="*60)
embedding = model.encode(["测试文本"])
print(f"词袋模型：词汇表大小 = 向量维度（通常10万+）")
print(f"词嵌入模型：固定维度 = {embedding.shape[1]} 维")
print(f"向量类型：稠密向量（每个维度都有值）")

print("\n" + "="*60)
print("思考题")
print("="*60)
print("1. 为什么词嵌入能识别'苹果'和'香蕉'相似？")
print("2. 这些向量是怎么训练出来的？")
print("3. 为什么300维就能表示所有词的语义？")
