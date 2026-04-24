"""
分布式假设的直观演示
展示为什么"苹果"和"香蕉"会被认为相似
"""

print("="*70)
print("分布式假设：通过上下文理解词义")
print("="*70)

# 模拟训练语料
corpus = [
    "我 喜欢 吃 苹果",
    "我 喜欢 吃 香蕉",
    "我 喜欢 吃 橙子",
    "苹果 很 甜",
    "香蕉 很 甜",
    "橙子 很 甜",
    "我 讨厌 吃 青菜",
    "青菜 很 苦",
    "我 开 汽车",
    "汽车 很 快",
]

print("\n训练语料（10个句子）:")
for i, sent in enumerate(corpus, 1):
    print(f"  {i}. {sent}")

# 统计每个词的上下文
def get_context(word, corpus):
    """获取一个词的所有上下文词"""
    contexts = {"left": [], "right": []}
    for sent in corpus:
        words = sent.split()
        if word in words:
            idx = words.index(word)
            if idx > 0:
                contexts["left"].append(words[idx-1])
            if idx < len(words) - 1:
                contexts["right"].append(words[idx+1])
    return contexts

# 分析几个关键词
target_words = ["苹果", "香蕉", "青菜", "汽车"]

print("\n" + "="*70)
print("步骤1：统计每个词的上下文")
print("="*70)

word_contexts = {}
for word in target_words:
    contexts = get_context(word, corpus)
    word_contexts[word] = contexts
    print(f"\n'{word}' 的上下文:")
    print(f"  左边的词: {contexts['left']}")
    print(f"  右边的词: {contexts['right']}")

# 计算上下文相似度
def context_similarity(word1, word2, word_contexts):
    """计算两个词的上下文相似度"""
    ctx1 = word_contexts[word1]
    ctx2 = word_contexts[word2]

    # 合并左右上下文
    all_ctx1 = set(ctx1['left'] + ctx1['right'])
    all_ctx2 = set(ctx2['left'] + ctx2['right'])

    # 计算交集和并集
    intersection = all_ctx1 & all_ctx2
    union = all_ctx1 | all_ctx2

    # Jaccard相似度
    if len(union) == 0:
        return 0
    return len(intersection) / len(union)

print("\n" + "="*70)
print("步骤2：计算上下文相似度（Jaccard相似度）")
print("="*70)

print("\n相似度矩阵:")
print(f"{'':8}", end="")
for w in target_words:
    print(f"{w:8}", end="")
print()

for w1 in target_words:
    print(f"{w1:8}", end="")
    for w2 in target_words:
        sim = context_similarity(w1, w2, word_contexts)
        print(f"{sim:8.2f}", end="")
    print()

print("\n" + "="*70)
print("关键发现")
print("="*70)

# 计算具体的相似度
sim_apple_banana = context_similarity("苹果", "香蕉", word_contexts)
sim_apple_vegetable = context_similarity("苹果", "青菜", word_contexts)
sim_apple_car = context_similarity("苹果", "汽车", word_contexts)

print(f"\n'苹果' vs '香蕉': 相似度 = {sim_apple_banana:.2f}")
print(f"  原因: 它们的上下文高度重合")
print(f"  共同上下文: 吃、喜欢、很、甜")

print(f"\n'苹果' vs '青菜': 相似度 = {sim_apple_vegetable:.2f}")
print(f"  原因: 有部分共同上下文（都能'吃'）")
print(f"  但情感不同（甜 vs 苦）")

print(f"\n'苹果' vs '汽车': 相似度 = {sim_apple_car:.2f}")
print(f"  原因: 上下文完全不同")
print(f"  一个是食物，一个是交通工具")

print("\n" + "="*70)
print("这就是分布式假设的核心思想！")
print("="*70)
print("1. 统计每个词的上下文")
print("2. 上下文相似 → 词义相似")
print("3. 神经网络自动学习这种模式")
print("4. 最终得到能表示语义的向量")

print("\n" + "="*70)
print("Word2Vec 的训练目标")
print("="*70)
print("给定上下文，预测中心词（CBOW）:")
print("  输入: [我, 喜欢, 吃, ___, 很, 甜]")
print("  输出: 苹果 (或 香蕉、橙子)")
print()
print("或者，给定中心词，预测上下文（Skip-gram）:")
print("  输入: 苹果")
print("  输出: [吃, 甜, 喜欢, ...]")
print()
print("训练过程中，能互相预测的词，向量会越来越接近！")
