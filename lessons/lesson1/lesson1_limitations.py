"""
演示词袋模型的语义盲区
"""
import numpy as np

def text_to_vector(text, vocab):
    """简单的词袋向量化"""
    vector = np.zeros(len(vocab))
    for char in text:
        if char in vocab:
            idx = vocab.index(char)
            vector[idx] += 1
    return vector

def cosine_similarity(v1, v2):
    """余弦相似度"""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

print("="*60)
print("词袋模型的语义盲区")
print("="*60)

# 场景1：近义词
texts1 = [
    "我喜欢吃苹果",
    "我爱吃苹果",
    "我讨厌吃苹果"
]

vocab1 = list(set(''.join(texts1)))
vectors1 = [text_to_vector(t, vocab1) for t in texts1]

print("\n场景1：近义词问题")
print("-"*60)
sim_like_love = cosine_similarity(vectors1[0], vectors1[1])
sim_like_hate = cosine_similarity(vectors1[0], vectors1[2])

print(f"'我喜欢吃苹果' vs '我爱吃苹果':   相似度 {sim_like_love:.4f}")
print(f"'我喜欢吃苹果' vs '我讨厌吃苹果': 相似度 {sim_like_hate:.4f}")
print("\n问题：'喜欢'和'爱'是近义词，但相似度不高")
print("     '喜欢'和'讨厌'是反义词，但相似度也不低！")

# 场景2：同类词
texts2 = [
    "我想吃苹果",
    "我想吃香蕉",
    "我想吃汽车"  # 荒谬的句子
]

vocab2 = list(set(''.join(texts2)))
vectors2 = [text_to_vector(t, vocab2) for t in texts2]

print("\n场景2：同类词问题")
print("-"*60)
sim_apple_banana = cosine_similarity(vectors2[0], vectors2[1])
sim_apple_car = cosine_similarity(vectors2[0], vectors2[2])

print(f"'我想吃苹果' vs '我想吃香蕉': 相似度 {sim_apple_banana:.4f}")
print(f"'我想吃苹果' vs '我想吃汽车': 相似度 {sim_apple_car:.4f}")
print("\n问题：苹果和香蕉都是水果，应该相似")
print("     但系统认为'吃苹果'和'吃汽车'一样相似！")

# 场景3：语序问题
texts3 = [
    "猫咬了狗",
    "狗咬了猫"
]

vocab3 = list(set(''.join(texts3)))
vectors3 = [text_to_vector(t, vocab3) for t in texts3]

print("\n场景3：语序问题")
print("-"*60)
sim_order = cosine_similarity(vectors3[0], vectors3[1])
print(f"'猫咬了狗' vs '狗咬了猫': 相似度 {sim_order:.4f}")
print("\n问题：意思完全相反，但相似度是1.0（完全相同）！")
print("     因为词袋模型忽略了词的顺序")

print("\n" + "="*60)
print("总结：词袋模型的三大缺陷")
print("="*60)
print("1. 无法理解语义：'喜欢'≠'爱'，'苹果'≠'香蕉'")
print("2. 无法区分情感：'喜欢'和'讨厌'被同等对待")
print("3. 忽略语序：'猫咬狗'='狗咬猫'")
print("\n这就是为什么我们需要词嵌入（Word Embedding）！")
