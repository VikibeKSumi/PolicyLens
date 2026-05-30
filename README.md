---
title: PolicyLens
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---


# PolicyLens

**Two-stage retrieval with modular design and centralized configuration**  

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![GPU](https://img.shields.io/badge/GPU-4GB_VRAM-green)
![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-orange)
![Groq](https://img.shields.io/badge/LLM-Groq-purple)

---

### ✨ Key Features

- **Two-stage retrieval**: Bi-encoder and Cross-encoder
- **Graph agent configuration**: Each pipeline is a graph node.
- **Semantic cache**: Cache response based on meaning
- **Query rewriting**: Rewrites user query for accurate retrieval
- **Context compression**: Slices word count sent to LLM
- **GPU acceleration**: For fast processing of embedding and reranking
- **Cloud vectorDB**: To store embedded document vectors
- **Cloud persistent cache**: To store cache permanently
- **Hashing**: Check file duplicates to prevent re-ingestion
- **Indic text normalization**: To analyse Hindi language in govt. docs
- **Data collection**: Collected datas by scrapping webpages
- **Evaluation**: Ragas evaluation on 50 golden dataset
- **Logging**: Console logging to monitor processes
- **Testing**: Unit test for cache, compression, rerank, retrieve
- **Centralized configuration**: Single configuration with zero duplication
- **Modular design**: Each component separated by its concern
- **API exposure**: API enabled for accessiblity
- **API authentication**: To prevent unauthorized access from outside
- **Docker compose**: Multiple docker image orchestration
- **Cloud host**: Deployed the application to cloud
---

### 🛠 Tech Stack

- **Embedding**: 'BAAI/bge-small-en-v1.5'
- **Reranker**: 'BAAI/bge-reranker-base'
- **LLM**: 'llama-3.3-70b-versatile'
- **Framework**: LlamaIndex
- **Vector store**: Qdrant cloud
- **Cache store**: Redis
- **Evaluation**: RAGAS + Langchain
- **Cloud deploy**: Huggingface spaces

---

### 📈 Current Performance (13 Golden dataset evaluation)

- **Faithfulness**: 0.96

---

### 🚀 Quick Start

# 1. Clone the repository
```bash
    git clone https://github.com/VikibeKSumi/PolicyLens.git
    cd PolicyLens
```
# 2. (optional but recommended) Create and activate environment
```bash
    conda create -n policylens python=3.10 -y
    conda activate policylens
```

# 3. Install dependencies
```bash
    pip install -r requirements.txt
```

# 4. Add your Groq API key
(Create a file named .env in the root folder and put this line inside it)
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  <- Replace with your Groq API
```

# 5. Put your PDF files
(budget, economic survey provided as default inside this folder)
```text
    data/raw_docs/
```

# 6. Ingest the documents
```bash
    python ingestion.py
```

# 7. Run Qdrant server
(install docker have it running)
```bash
docker run -d -p 6333:6333 -v $(pwd)/data/qdrant_storage:/qdrant/storage qdrant/qdrant

```
# 8. (main) Run the RAG Engine
```bash
    python run_graph.py
```

# 9. (optional but recommended) Run ragas evaluation
```bash
    python evaluation.py
```

---

### 📁 Folder Structure

```text
policylens/
|── run_graph.py                          # Main CLI entry point
|── ingestion.py                    # Document ingestion
|── evaluation.py                   # RAGAS evaluation
|── src/
|   |── core/
|   |   |── schemas.py
|   |   |── text_utils.py
|   |── pipeline/
|   |   |── cache.py
|   |   |── compression.py
|   |   |── query_rewriter.py
|   |   |── reranker.py
|   |   |── retrieval.py
|   |── services/     
|   |   |── embedder.py
|   |   |── llm.py
|   |   |── vector_store.py    
|   |── graph.py
|   |── state.py
|── config/
|   |── config.py
|   |── settings.yaml
|── data/
```

---

### 🚧 Future Improvements
- Graph based knowledge

---
