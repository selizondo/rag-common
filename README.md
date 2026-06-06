# rag-common

![Tests](https://github.com/selizondo/rag-common/actions/workflows/ci.yml/badge.svg)

Shared library for the RAG pipeline projects. Chunkers, IR metrics, FAISS vector store, and hybrid retrieval: these primitives are identical across downstream projects, so they live in one place where an IR metric bug is fixed once and chunk IDs never collide between chunking configurations.

**Stack:** Python · FAISS · rank-bm25 · NLTK · Pydantic

## What It Provides

### IR metrics with hand-verified math

`recall_at_k`, `precision_at_k`, `ndcg_at_k`, `mrr`, `map_score` — all operating on chunk ID strings, independent of any embedding library. 143 tests, including parametrized edge cases and hand-computed expected values. **Precision@K is correctly capped at 1/K for single-ground-truth queries** (max 0.20 at K=5): this is TREC/BEIR behavior, not a bug. MRR and Recall@K are the primary signals for this evaluation setup.

### Score fusion that doesn't let BM25 dominate

BM25 scores are unbounded floats. Dense scores are cosine similarities in [-1, 1]. Fusing without normalization makes alpha meaningless: a BM25 score of 12.4 overwhelms a dense score of 0.31, so you ask for 60% dense and get 90% dense in practice. `HybridRetriever` min-max normalizes both sides to [0, 1] independently before fusing, then applies alpha as its intended blend. The tradeoffs.md covers when to use RRF instead.

### Protocol-based adapters

`VectorStoreProtocol` and `RetrieverProtocol` are structural. Any class with the right methods qualifies without inheritance. Swap FAISS for Qdrant: add one file, change zero lines of retrieval code.

### Three chunkers, one interface

`FixedSizeChunker`, `SentenceBasedChunker`, `SemanticChunker` — all expose `.chunk(text, metadata={}) -> list[Chunk]`. Project-specific chunkers in each downstream repo follow the same interface.

**Used by:**
- [rag-pipeline-systematic-evals](https://github.com/selizondo/rag-pipeline-systematic-evals): single-PDF grid search with synthetic QA evaluation
- [rag-pipeline-experimentation](https://github.com/selizondo/rag-pipeline-experimentation): multi-paper QA assistant with real qrels, citations, and Streamlit UI

---

## Go Deeper

| Audience | Doc |
|----------|-----|
| Running the code | [Setup and Usage](docs/setup.md) |
| Engineering decisions | [Design and Tradeoffs](docs/engineering.md) |
| What breaks and why | [Failure Modes](docs/failures.md) |
