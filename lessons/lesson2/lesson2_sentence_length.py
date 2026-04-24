"""
验证：句子长度对语义理解的影响
解答学生的疑问：为什么"吃汽车"比"吃香蕉"相似度还高？
"""
from sentence_transformers import SentenceTransformer
import numpy as np

print("加载模型...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_similarity(text1, text2):
    embeddings = model.encode([text1, text2])
    cos_sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return cos_sim

print("\n" + "="*70)
print("实验：句子长度的影响")
print("="*70)

# 测试1：极短句子（原始问题）
print("\n【测试1：极短句子（4个字）】")
pairs_short = [
    ("我想吃苹果", "我想吃香蕉"),
    ("我想吃苹果", "我想吃汽车"),
]

for text1, text2 in pairs_short:
    sim = get_similarity(text1, text2)
    print(f"'{text1}' vs '{text2}'")
    print(f"  相似度: {sim:.4f}")

# 测试2：中等长度句子
print("\n【测试2：中等长度句子（增加上下文）】")
pairs_medium = [
    ("我今天很想吃一个新鲜的苹果", "我今天很想吃一个新鲜的香蕉"),
    ("我今天很想吃一个新鲜的苹果", "我今天很想吃一个新鲜的汽车"),
]

for text1, text2 in pairs_medium:
    sim = get_similarity(text1, text2)
    print(f"'{text1}' vs '{text2}'")
    print(f"  相似度: {sim:.4f}")

# 测试3：更长的句子
print("\n【测试3：长句子（更多语义信息）】")
pairs_long = [
    ("苹果是一种营养丰富的水果，味道香甜可口",
     "香蕉是一种营养丰富的水果，味道香甜可口"),
    ("苹果是一种营养丰富的水果，味道香甜可口",
     "汽车是一种交通工具，可以快速移动"),
]

for text1, text2 in pairs_long:
    sim = get_similarity(text1, text2)
    print(f"'{text1}' vs '{text2}'")
    print(f"  相似度: {sim:.4f}")

print("\n" + "="*70)
print("结论")
print("="*70)
print("1. 句子越短，结构相似度的影响越大")
print("2. 句子越长，语义差异越明显")
print("3. 极短句子（4-5字）可能出现反直觉的结果")
print("4. 实际应用中，建议使用完整的句子或段落")

print("\n" + "="*70)
print("为什么会这样？")
print("="*70)
print("短句子: '我想吃X'")
print("  - 75%的内容是相同的（我想吃）")
print("  - 只有25%不同（X）")
print("  - 模型主要看到了结构相似性")
print()
print("长句子: '苹果是一种营养丰富的水果...'")
print("  - 包含更多语义信息（水果、营养、香甜）")
print("  - 模型能更好地区分语义差异")

print("\n" + "="*70)
print("面试要点")
print("="*70)
print("Q: 为什么短句子的语义相似度不准确？")
print("A: 因为短句子中结构占比大，语义信息少。")
print("   模型主要捕捉到了句法结构的相似性。")
print()
print("Q: 如何提高短文本的检索准确率？")
print("A: 1. 使用查询扩展（Query Expansion）")
print("   2. 结合关键词匹配（混合检索）")
print("   3. 使用专门针对短文本优化的模型")
