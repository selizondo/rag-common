# Failure Scenarios

Documented failure modes with detection mechanisms and fallback behavior.

---

## Failure 1: Precision@K Appears Broken at Low Values

### What breaks
With single-ground-truth queries (one relevant chunk per question), `Precision@K` is mathematically capped at `1/K`. At K=5 this means max 0.20. Callers expecting 0.0–1.0 range interpret low scores as a bug.

### Detection mechanism
This is expected behavior, not a defect. `rag_common/metrics.py` uses standard TREC/BEIR Precision@K. The README documents the cap. Use MRR and Recall@K as primary signals for single-ground-truth evaluation.

### Fallback behavior
No code fix needed. If multi-relevance evaluation is required, pass sets with multiple ground-truth chunk IDs per query.

---

## Failure 2: SemanticChunker Never Splits When Embeddings Are Flat

### What breaks
`SemanticChunker` splits on cosine-similarity drops between adjacent sentences. If the embedding model returns near-identical vectors for all sentences (e.g., a zero-variance test stub), no drops occur and the entire document becomes one chunk.

### Detection mechanism
`max_sentences` caps chunk size even when similarity never drops below `breakpoint_threshold`. If `max_sentences` is set, the chunker always produces bounded-size output regardless of similarity values.

### Fallback behavior
Always set `max_sentences` in production. Use `InMemoryVectorStore` with a real embedding model in integration tests — a zero-variance stub won't exercise split logic.

---

## Failure 3: FAISS Index Size Mismatch After Load

### What breaks
`FAISSVectorStore.load()` restores the index from disk. If the embedding dimension at load time differs from the saved index (e.g., switching from `all-MiniLM-L6-v2` 384-dim to `text-embedding-ada-002` 1536-dim), FAISS raises a dimension mismatch error on the first search.

### Detection mechanism
FAISS raises `AssertionError` or `RuntimeError` with a dimension message on the first `search()` call after loading a mismatched index.

### Fallback behavior
Delete and rebuild the index when changing embedding models. The index filename convention in downstream projects includes the model name (e.g., `faiss_minilm_384.index`) to make mismatches visible before load.

---

## Failure 4: HybridRetriever Score Fusion Collapses When One Retriever Returns All-Zero Scores

### What breaks
`HybridRetriever` min-max normalises BM25 and dense scores independently. If all BM25 candidates score 0 (no keyword overlap), min-max normalization produces `0/0 = NaN` and fused scores become undefined.

### Detection mechanism
`HybridRetriever` guards against zero-range normalisation: if `max - min == 0`, scores are set to 0.5 (neutral contribution). The guard is in `_normalize()` in `retrievers.py`.

### Fallback behavior
The guard handles it silently. If BM25 consistently produces zero scores on a corpus, consider whether BM25 is useful for that data — semantic-only retrieval (`alpha=1.0`) may be more appropriate.
