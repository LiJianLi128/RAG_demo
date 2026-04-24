# RAG 入门项目

一个简单实用的 RAG（Retrieval-Augmented Generation）系统实现，适合学习和实验。

## 功能特点

- 基于 FAISS 的本地向量存储
- 使用 sentence-transformers 进行文本嵌入
- 支持文档分块和检索
- 可选接入 OpenAI API 进行答案生成
- 向量存储持久化

## 快速开始

### 1. 安装依赖

```bash
source venv/Scripts/activate
pip install -r requirements.txt
```

### 2. 运行基础示例

```bash
source venv/Scripts/activate
python examples/example_basic.py
```

这个示例会：
- 添加一些关于 RAG 的示例文档
- 执行几个查询
- 展示检索结果

### 3. 使用自己的文档

创建数据目录和示例文件：

```bash
mkdir data
echo "你的文档内容" > data/sample.txt
```

然后运行：

```bash
source venv/Scripts/activate
python examples/example_file.py
```

## 配置 OpenAI（可选）

如果想使用 LLM 生成答案而不只是检索：

1. 复制配置文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的 API Key：
```
OPENAI_API_KEY=your_api_key_here
```

3. 修改代码中的 `use_openai=True`

## 项目结构

```
.
├── config.py
├── document_processor.py
├── vector_store.py
├── rag_system.py
├── examples/
│   ├── example_basic.py
│   └── example_file.py
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
├── requirements.txt
└── README.md
```

## 核心概念

### 1. 文档分块（Chunking）
将长文档切分成小块，便于检索和处理。可在 `config.py` 中调整：
- `CHUNK_SIZE`: 每块大小
- `CHUNK_OVERLAP`: 块之间重叠部分

### 2. 文本嵌入（Embedding）
将文本转换为向量表示。使用的模型：
- `paraphrase-multilingual-MiniLM-L12-v2`（支持中文）

### 3. 向量检索
使用 FAISS 进行高效的相似度搜索，找到最相关的文档块。

### 4. 答案生成（可选）
将检索到的文档作为上下文，使用 LLM 生成答案。

## 下一步学习

1. 尝试不同的分块策略
2. 实验不同的嵌入模型
3. 调整检索参数（Top-K）
4. 学习 `lessons/lesson6/LANGCHAIN_THEORY.md`
5. 运行 `lessons/lesson6/lesson6_langchain_retrieval.py`
6. 观察并优化检索效果
7. 再接入 LLM API 完成生成阶段

## 常见问题

**Q: 安装 LangChain 依赖时报版本冲突？**
A: 当前 LangChain 相关包要求较新的 `numpy` 和 `sentence-transformers`，如果你在旧依赖基础上扩展，需要同步升级这两个包。

**Q: 首次运行很慢？**
A: 第一次运行会下载嵌入模型（约 400MB），之后会缓存。

**Q: 不想使用 OpenAI？**
A: 完全可以！设置 `use_openai=False`，系统会返回检索到的原始文档。

**Q: 如何处理 PDF 文件？**
A: 可以添加 `PyPDF2` 或 `pdfplumber` 库来解析 PDF。
