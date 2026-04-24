# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) learning project with a progressive tutorial structure. It contains both a complete RAG implementation and step-by-step lessons for understanding the fundamentals.

## Virtual Environment Setup

**IMPORTANT**: Always use the virtual environment to avoid dependency conflicts.

```bash
# Activate virtual environment (required before running any Python code)
source venv/Scripts/activate

# First time setup: Install dependencies
pip install -r requirements.txt
# Note: First run will download ~400MB embedding model
```

## Running Examples

**Always activate venv first!**

```bash
# Activate virtual environment
source venv/Scripts/activate

# Run basic example (uses in-memory documents)
python examples/example_basic.py

# Run file-based example (loads from data/sample.txt)
python examples/example_file.py
```

## Running Lessons

**Lesson 1: Vector Retrieval Basics**
```bash
source venv/Scripts/activate
python lessons/lesson1/lesson1_simple_vector.py        # Basic bag-of-words
python lessons/lesson1/lesson1_cosine_explained.py     # Cosine similarity deep dive
python lessons/lesson1/lesson1_limitations.py          # Limitations of bag-of-words
```

**Lesson 2: Word Embeddings**
```bash
source venv/Scripts/activate
python lessons/lesson2/lesson2_embeddings.py           # Semantic understanding
python lessons/lesson2/lesson2_cross_lingual.py        # Cross-lingual retrieval
```

## Configuration

The system works without OpenAI by default (retrieval-only mode). To enable LLM generation:

1. Copy `.env.example` to `.env`
2. Add your `OPENAI_API_KEY`
3. Set `use_openai=True` when initializing `RAGSystem`

All configuration parameters are in `config.py`:
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: Document chunking strategy
- `TOP_K`: Number of documents to retrieve
- `EMBEDDING_MODEL`: sentence-transformers model name
- `VECTOR_STORE_PATH`: Persistent storage location

## Architecture

The codebase follows a modular RAG pipeline:

1. **document_processor.py**: Text loading and chunking
   - `load_text_file()`: UTF-8 file loading
   - `chunk_text()`: Sliding window chunking with overlap

2. **vector_store.py**: Embedding and retrieval layer
   - Uses sentence-transformers for embeddings
   - FAISS IndexFlatL2 for vector search (L2 distance)
   - Persists to disk: `index.faiss` + `documents.pkl`

3. **rag_system.py**: High-level orchestration
   - Combines retrieval + generation
   - Falls back to retrieval-only if no LLM configured
   - Handles document ingestion and querying

4. **lesson*.py**: Educational examples
   - Progressive complexity from basic concepts to full implementation
   - Designed for interactive learning with thought exercises

## Key Implementation Details

**Vector Store Behavior:**
- First call to `add_documents()` initializes FAISS index with embedding dimension
- Subsequent calls append to existing index
- `save()` must be called explicitly to persist
- `load()` returns `True` if existing store found, `False` otherwise

**Distance Metric:**
- Uses L2 (Euclidean) distance in FAISS, not cosine similarity
- Lower distance = more similar (opposite of cosine)
- Results are `(document, distance)` tuples

**Chunking Strategy:**
- Character-based with word boundary detection
- Overlap prevents context loss at chunk boundaries
- Simple implementation - may split mid-word for non-space languages

## Lesson Structure

This is a teaching codebase. The lesson files demonstrate concepts progressively:
- `lessons/lesson1/lesson1_simple_vector.py`: Bag-of-words vector retrieval from scratch
- Future lessons will cover embeddings, FAISS internals, optimization, etc.

When working with lessons, preserve the pedagogical structure and explanatory comments.

## Important Reminders

**Virtual Environment**: This project uses a virtual environment (`venv/`). Always remind users to activate it:
```bash
source venv/Scripts/activate
```

When providing commands to run, always include the activation step first.

## Teaching Mode Instructions

This project includes an interactive teaching component tracked in `TUTORIAL.md` and `LESSON*.md` files. When continuing the tutorial:

1. **Read the current state**: Check `TUTORIAL.md` to understand:
   - Which lesson is in progress
   - What concepts have been covered
   - Student's answers to previous questions
   - Current learning progress

2. **Adapt teaching approach**: Based on student responses:
   - If answers show deep understanding → move faster, add advanced topics
   - If answers show confusion → slow down, add more examples
   - If answers show partial understanding → provide targeted clarification

3. **Update documentation dynamically**:
   - Add student answers to `TUTORIAL.md` under "学生回答"
   - Update lesson status (进行中 ⏳ / 已完成 ✓ / 待开始 ⏸️)
   - Create new `lessonN_*.py` files as needed
   - Expand "面试要点" based on student questions
   - Add "常见误区" section if student makes typical mistakes

4. **Maintain pedagogical quality**:
   - Always explain WHY before HOW
   - Provide multiple perspectives (math, intuition, code)
   - Include real-world implications and trade-offs
   - Connect concepts to interview questions
   - Use concrete examples before abstractions

5. **Progressive complexity**:
   - Each lesson builds on previous ones
   - Start with simple, working code
   - Gradually reveal complexity and edge cases
   - End each lesson with thought-provoking questions

6. **Assessment and feedback**:
   - Require student to answer questions before proceeding
   - Provide detailed feedback on answers
   - Adjust next lesson difficulty based on performance
   - Track misconceptions and address them explicitly
