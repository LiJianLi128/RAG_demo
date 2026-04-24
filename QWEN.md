# RAG 学习项目 - QWEN 上下文

## 项目概述

这是一个 **RAG（Retrieval-Augmented Generation，检索增强生成）** 的入门学习和实践项目。项目主要目标是：

1. **教学用途**：通过5课时的系统化学习，掌握 RAG 技术的核心原理和实现方法
2. **实践实现**：提供一个简单实用的 RAG 系统实现，适合学习和实验
3. **技术栈**：基于 Python，使用 FAISS 向量存储和 sentence-transformers 嵌入模型

## 项目结构

```
D:\myItem\Leaning\RAG\
├── 核心模块
│   ├── config.py              # 配置文件（分块大小、模型选择等）
│   ├── document_processor.py  # 文档处理模块（加载和分块）
│   ├── vector_store.py        # 向量存储模块（FAISS 索引管理）
│   └── rag_system.py          # RAG 核心模块（整合检索和生成）
│
├── examples/
│   ├── example_basic.py       # 基础使用示例（不使用 LLM）
│   └── example_file.py        # 从文件加载文档的示例
│
├── lessons/
│   ├── lesson1/
│   │   ├── lesson1_simple_vector.py
│   │   ├── lesson1_cosine_explained.py
│   │   └── lesson1_limitations.py
│   ├── lesson2/
│   │   ├── LESSON2.md
│   │   ├── LESSON2_THEORY.md
│   │   ├── lesson2_embeddings.py
│   │   ├── lesson2_cross_lingual.py
│   │   ├── lesson2_distribution_hypothesis.py
│   │   └── lesson2_sentence_length.py
│   ├── lesson3/
│   │   └── LESSON3_THEORY.md
│   ├── lesson4/
│   │   └── LESSON4_THEORY.md
│   ├── lesson5/
│   │   └── LESSON5_THEORY.md
│   └── lesson6/
│       ├── LANGCHAIN_THEORY.md
│       └── lesson6_langchain_retrieval.py
│
├── 数据和存储
│   ├── data/                  # 待处理的文档目录
│   └── vector_store/          # FAISS 向量存储持久化目录
│
└── 配置文件
    ├── requirements.txt       # Python 依赖
    ├── .env.example           # 环境变量模板
    └── README.md              # 项目说明
```

## 技术栈和依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| sentence-transformers | 2.2.2 | 文本嵌入模型 |
| faiss-cpu | 1.7.4 | 向量相似度搜索 |
| numpy | 1.24.3 | 数值计算 |
| openai | 1.12.0 | LLM API（可选） |
| python-dotenv | 1.0.0 | 环境变量管理 |

**嵌入模型**：`paraphrase-multilingual-MiniLM-L12-v2`（支持多语言，包括中文）

## 核心概念

### 1. 文档分块（Chunking）
- 将长文档切分成固定大小的块（默认 500 字符）
- 块之间有重叠（默认 50 字符）以保持上下文连贯
- 可在 `config.py` 中调整 `CHUNK_SIZE` 和 `CHUNK_OVERLAP`

### 2. 文本嵌入（Embedding）
- 将文本转换为高维向量表示
- 使用 sentence-transformers 库
- 向量能够捕捉语义信息

### 3. 向量检索
- 使用 FAISS 进行高效的相似度搜索
- 默认返回 Top-K（K=3）最相关的文档块

### 4. 答案生成（可选）
- 可将检索结果作为上下文，使用 OpenAI API 生成答案
- 设置 `use_openai=False` 则只返回检索结果

## 虚拟环境设置

**重要**：始终使用虚拟环境以避免依赖冲突。

```bash
# 激活虚拟环境（运行任何 Python 代码前必须执行）
venv\Scripts\activate

# 首次安装依赖
pip install -r requirements.txt
# 注意：首次运行会下载约 400MB 的嵌入模型
```

## 运行示例

**必须先激活虚拟环境！**

```bash
# 激活虚拟环境
venv\Scripts\activate

# 运行基础示例（使用内存中的文档）
python examples/example_basic.py

# 运行文件示例（从 data/sample.txt 加载）
python examples/example_file.py
```

## 运行课程脚本

**第一课：向量检索基础**
```bash
venv\Scripts\activate
python lessons/lesson1/lesson1_simple_vector.py        # 词袋模型基础
python lessons/lesson1/lesson1_cosine_explained.py     # 余弦相似度深入
python lessons/lesson1/lesson1_limitations.py          # 词袋模型的局限性
```

**第二课：词嵌入**
```bash
venv\Scripts\activate
python lessons/lesson2/lesson2_embeddings.py           # 语义理解
python lessons/lesson2/lesson2_cross_lingual.py        # 跨语言检索
```

## 配置

系统默认以检索模式运行（不使用 OpenAI）。要启用 LLM 生成功能：

1. 复制 `.env.example` 到 `.env`
2. 添加你的 `OPENAI_API_KEY`
3. 初始化 `RAGSystem` 时设置 `use_openai=True`

所有配置参数都在 `config.py` 中：
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: 文档分块策略
- `TOP_K`: 检索文档数量
- `EMBEDDING_MODEL`: sentence-transformers 模型名称
- `VECTOR_STORE_PATH`: 持久化存储路径

## 架构设计

代码库遵循模块化的 RAG 流水线设计：

1. **document_processor.py**: 文本加载和分块
   - `load_text_file()`: UTF-8 文件读取
   - `chunk_text()`: 滑动窗口分块，带重叠

2. **vector_store.py**: 嵌入和检索层
   - 使用 sentence-transformers 进行嵌入
   - FAISS IndexFlatL2 进行向量搜索（L2 距离）
   - 持久化到磁盘：`index.faiss` + `documents.pkl`

3. **rag_system.py**: 高层编排
   - 组合检索 + 生成
   - 如果没有 LLM 配置，回退到纯检索模式
   - 处理文档摄入和查询

4. **lesson*.py**: 教学示例
   - 从基础概念到完整实现的渐进式复杂度
   - 设计用于交互式学习，包含思考练习

## 关键实现细节

**向量存储行为：**
- 首次调用 `add_documents()` 时，用嵌入维度初始化 FAISS 索引
- 后续调用会追加到现有索引
- `save()` 必须显式调用才能持久化
- `load()` 返回 `True`（找到现有存储）或 `False`（无）

**距离度量：**
- 使用 L2（欧氏）距离，而非余弦相似度
- 距离越低 = 越相似（与余弦相似度相反）
- 结果格式为 `(document, distance)` 元组

**分块策略：**
- 基于字符，带单词边界检测
- 重叠防止在块边界丢失上下文
- 简单实现 - 对无空格语言（如中文）可能会在词中间切断

## 教学模式说明

这是一个教学代码库。课程文件按渐进式展示概念：
- `lessons/lesson1/lesson1_simple_vector.py`: 从零实现的词袋向量检索
- 后续课程将覆盖嵌入、FAISS 内部、优化等

处理课程文件时，请保持教学结构和解释性注释。

## 教学助手指南

本项目包含交互式教学组件，记录在 `TUTORIAL.md` 和 `LESSON*.md` 文件中。继续教程时：

1. **阅读当前状态**：检查 `TUTORIAL.md` 了解：
   - 正在进行的课程
   - 已覆盖的概念
   - 学生对之前问题的回答
   - 当前学习进度

2. **调整教学方法**：根据学生回答：
   - 如果回答显示深入理解 → 加快速度，添加高级话题
   - 如果回答显示困惑 → 放慢节奏，增加更多示例
   - 如果回答显示部分理解 → 针对性澄清

3. **动态更新文档**：
   - 将学生回答添加到 `TUTORIAL.md` 的"学生回答"部分
   - 更新课程状态（进行中 ⏳ / 已完成 ✓ / 待开始 ⏸️）
   - 根据需要创建新的 `lessonN_*.py` 文件
   - 根据学生问题扩展"面试要点"
   - 如果学生犯典型错误，添加"常见误区"部分

4. **保持教学质量**：
   - 始终先解释 WHY 再解释 HOW
   - 提供多重视角（数学、直觉、代码）
   - 包含现实世界的含义和权衡
   - 将概念与面试问题关联
   - 使用具体示例再引入抽象

5. **渐进式复杂度**：
   - 每课建立在之前的基础上
   - 从简单、可工作的代码开始
   - 逐渐揭示复杂性和边缘情况
   - 每课结束提出发人深省的问题

6. **评估和反馈**：
   - 要求学生回答问题后再继续
   - 对回答提供详细反馈
   - 根据表现调整下节课难度
   - 追踪误解并明确解决

## 课程大纲

| 课时 | 主题 | 状态 |
|------|------|------|
| 第一课 | 建立直觉 - 什么是向量检索（词袋模型、余弦相似度） | ✅ 完成 |
| 第二课 | 深入原理 - 词嵌入与语义理解（Word2Vec, BERT, sentence-transformers） | ✅ 完成 |
| 第三课 | 实战 - 构建真实的向量存储（FAISS 工作原理） | ✅ 理论完成 |
| 第四课 | 整合 - 完整的 RAG 系统（分块策略、Prompt 工程） | ✅ 理论完成 |
| 第五课 | 优化 - 生产级考虑（评估方法、混合检索、重排序） | ✅ 理论完成 |

## 开发约定

- **代码风格**：使用 Python 类型提示（typing 模块）
- **文档字符串**：所有函数都有 docstring 说明参数和返回值
- **配置分离**：所有配置参数集中在 `config.py`
- **模块化设计**：核心功能分为三个独立模块（document_processor, vector_store, rag_system）
- **教学优先**：课程代码应保持解释性注释和渐进式复杂度

## 重要文件说明

### `rag_system.py` - RAG 核心
`RAGSystem` 类是整个系统的入口，整合了：
- 文档添加和分块
- 向量存储管理
- 检索和答案生成

关键方法：
- `add_documents()`: 添加文档
- `add_documents_from_file()`: 从文件加载并分块
- `query()`: 查询并获取答案（支持检索模式或生成模式）
- `save()/load()`: 持久化向量存储

### `vector_store.py` - 向量存储
`VectorStore` 类管理 FAISS 索引：
- `add_documents()`: 生成嵌入并添加到索引
- `search()`: 执行相似度搜索，返回 `(document, distance)` 元组
- `save()/load()`: 保存/加载 `index.faiss` 和 `documents.pkl`

### `document_processor.py` - 文档处理
- `load_text_file()`: 读取 UTF-8 文本文件
- `chunk_text()`: 滑动窗口分块，支持重叠和单词边界检测

### `config.py` - 配置中心
包含所有可调整的参数：
- 嵌入模型选择（默认支持多语言）
- 分块大小和重叠
- 检索数量（TOP_K）
- OpenAI API 配置

## 后续学习方向

根据 TUTORIAL.md 中的建议：
1. 尝试不同的分块策略
2. 实验不同的嵌入模型
3. 调整检索参数
4. 添加重排序（Reranking）
5. 实现多轮对话
6. 添加评估指标
7. 学习 LangChain 框架
