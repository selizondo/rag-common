# Setup and Usage

## Installation

```bash
# As an editable path install (downstream project's pyproject.toml)
[tool.uv.sources]
rag-common = { path = "../rag_common", editable = true }

[project]
dependencies = ["rag-common", ...]
```

Or directly:

```bash
pip install -e ../rag_common
```

## Tests

```bash
cd rag_common
pip install -e ".[dev]"
pytest tests/ -v
```

143 tests: metrics (hand-verified math), chunkers (deterministic stub embedder), vector store (parametrized against FAISS and InMemory), retrievers (Protocol checks + alpha boundary tests).

## Usage

### Metrics

```python
from rag_common import metrics

query_results = [
    (["chunk-a", "chunk-b", "chunk-c"], {"chunk-a"}),
    (["chunk-x", "chunk-a", "chunk-b"], {"chunk-a"}),
]
scores = metrics.evaluate(query_results, k=5)
# {"recall@5": 1.0, "precision@5": 0.2, "mrr": 0.75, "map": 0.75, "ndcg@5": 1.0}
```

### Chunkers

```python
from rag_common.chunkers import FixedSizeChunker, SentenceBasedChunker, SemanticChunker
from sentence_transformers import SentenceTransformer

chunks = FixedSizeChunker(chunk_size=512, overlap=64).chunk(text, metadata={"source": "paper.pdf"})
chunks = SentenceBasedChunker(sentences_per_chunk=5, overlap_sentences=1).chunk(text)

model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = SemanticChunker(embed_fn=model.encode, breakpoint_threshold=0.65).chunk(text)
```

### Vector Store

```python
from rag_common.vector_store import FAISSVectorStore, InMemoryVectorStore, VectorStoreProtocol

store: VectorStoreProtocol = FAISSVectorStore()     # production
store: VectorStoreProtocol = InMemoryVectorStore()  # tests

store.add(chunks, embeddings)          # embeddings: np.ndarray (N, D)
results = store.search(query_emb, top_k=5)
store.save("data/indices/my_index")
store.load("data/indices/my_index")
```

### Hybrid Retrieval

```python
from rag_common.retrievers import BM25Retriever, DenseRetriever, HybridRetriever

bm25   = BM25Retriever(chunks)
dense  = DenseRetriever(store, embed_fn=model.encode)
hybrid = HybridRetriever(dense, bm25, alpha=0.6)   # 60% dense, 40% BM25

results = hybrid.retrieve("what is backpropagation?", top_k=5)
```

## Module Summary

| Module | Role |
|--------|------|
| `rag_common.models` | `Chunk`, `RetrievalResult` Pydantic types |
| `rag_common.metrics` | IR evaluation: recall, precision, NDCG, MRR, MAP |
| `rag_common.chunkers` | `FixedSizeChunker`, `SentenceBasedChunker`, `SemanticChunker` |
| `rag_common.vector_store` | `FAISSVectorStore`, `InMemoryVectorStore`, `VectorStoreProtocol` |
| `rag_common.retrievers` | `BM25Retriever`, `DenseRetriever`, `HybridRetriever` |
