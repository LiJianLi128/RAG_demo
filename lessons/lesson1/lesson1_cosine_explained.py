"""
深入理解余弦相似度的计算
回答学生的疑问：为什么不是简单的占比平均？
"""
import numpy as np

print("="*60)
print("余弦相似度的真实计算过程")
print("="*60)

# 简化示例：只看关键的几个字
# 词汇表：['我', '吃', '苹', '果', '喜', '欢', '想']
#          0     1     2     3     4     5     6

# 查询："我想吃苹果" → 有：我(1), 想(1), 吃(1), 苹(1), 果(1)
query_vec = np.array([1, 1, 1, 1, 0, 0, 1])  # 5个字

# 文档1："我喜欢吃苹果" → 有：我(1), 喜(1), 欢(1), 吃(1), 苹(1), 果(1)
doc_vec = np.array([1, 1, 1, 1, 1, 1, 0])    # 6个字

print("\n查询向量:", query_vec)
print("文档向量:", doc_vec)

# 步骤1：计算点积（重合的维度相乘后求和）
dot_product = np.dot(query_vec, doc_vec)
print(f"\n步骤1：点积 = {dot_product}")
print("  解释：两个向量对应位置相乘后求和")
print("  1*1 + 1*1 + 1*1 + 1*1 + 0*1 + 0*1 + 1*0 = 4")
print("  → 有4个字同时出现在两个句子中")

# 步骤2：计算向量的长度（范数）
norm_query = np.linalg.norm(query_vec)
norm_doc = np.linalg.norm(doc_vec)
print(f"\n步骤2：计算向量长度")
print(f"  查询向量长度 = √(1²+1²+1²+1²+0²+0²+1²) = √5 = {norm_query:.4f}")
print(f"  文档向量长度 = √(1²+1²+1²+1²+1²+1²+0²) = √6 = {norm_doc:.4f}")

# 步骤3：计算余弦相似度
cosine_sim = dot_product / (norm_query * norm_doc)
print(f"\n步骤3：余弦相似度 = 点积 / (长度1 × 长度2)")
print(f"  = {dot_product} / ({norm_query:.4f} × {norm_doc:.4f})")
print(f"  = {dot_product} / {norm_query * norm_doc:.4f}")
print(f"  = {cosine_sim:.4f}")

print("\n" + "="*60)
print("关键理解")
print("="*60)
print("1. 点积反映了'重合程度'（共同字数）")
print("2. 除以长度是'归一化'（消除句子长度影响）")
print("3. 结果是向量夹角的余弦值，不是简单的占比")
print()
print("为什么不是占比平均？")
print("  - 占比平均：(3/5 + 4/6)/2 ≈ 0.63")
print(f"  - 余弦相似度：{cosine_sim:.4f}")
print("  - 差异原因：余弦考虑了向量空间的几何关系")

# 现在回答问题2
print("\n" + "="*60)
print("问题2：如果改成'我想吃香蕉'会怎样？")
print("="*60)

# 词汇表扩展：['我', '吃', '苹', '果', '喜', '欢', '想', '香', '蕉']
#              0     1     2     3     4     5     6     7     8

# 新查询："我想吃香蕉"
query_vec2 = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1])  # 5个字

# 文档1："我喜欢吃苹果"（在新词汇表中）
doc_vec2 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0])    # 6个字

dot_product2 = np.dot(query_vec2, doc_vec2)
norm_query2 = np.linalg.norm(query_vec2)
norm_doc2 = np.linalg.norm(doc_vec2)
cosine_sim2 = dot_product2 / (norm_query2 * norm_doc2)

print(f"\n新查询向量: {query_vec2}")
print(f"文档向量:   {doc_vec2}")
print(f"\n点积 = {dot_product2} (只有'我'和'吃'重合)")
print(f"查询长度 = √5 = {norm_query2:.4f}")
print(f"文档长度 = √6 = {norm_doc2:.4f}")
print(f"\n余弦相似度 = {cosine_sim2:.4f}")

print("\n对比：")
print(f"  '我想吃苹果' vs '我喜欢吃苹果': {cosine_sim:.4f}")
print(f"  '我想吃香蕉' vs '我喜欢吃苹果': {cosine_sim2:.4f}")
print(f"\n差异原因：")
print(f"  - 第一个：4个共同字（我、吃、苹、果）")
print(f"  - 第二个：2个共同字（我、吃）")
print(f"  - 相似度下降了 {(cosine_sim - cosine_sim2):.4f}")
