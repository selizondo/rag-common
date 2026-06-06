# Design and Tradeoffs

---

## Protocol-Based Adapters over ABCs

`VectorStoreProtocol` and `RetrieverProtocol` use structural protocols (`@runtime_checkable`), not abstract base classes. Any class that has the required methods qualifies without inheritance. Adding a Qdrant or Pinecone backend is a new file with zero changes to retrieval code.

Tradeoff: protocol violations are caught at type-check time (mypy), not at instantiation. A class that forgets `save()` won't fail until `save()` is called.

---

## FAISS IndexFlatIP with L2 Normalization

After L2-normalizing all embeddings at ingest time, inner product equals cosine similarity and scores land in [-1, 1]. This matches `HybridRetriever`'s expectation on the dense side of score fusion. The alternative, `IndexFlatL2`, returns squared Euclidean distances in [0, 4] and would need a separate normalization formula.

Scale boundary: `IndexFlatIP` performs exhaustive O(N) search. At approximately 1M chunks, it becomes too slow for interactive use. The upgrade path is `IndexIVFFlat` (approximate nearest-neighbor search) or a hosted ANN service. The `VectorStoreProtocol` interface does not change on the upgrade.

---

## Score Fusion: Linear Interpolation vs RRF

### The problem

BM25 scores are unbounded positive floats. Dense scores are cosine similarities in [-1, 1]. Fusing raw scores lets BM25 dominate: a BM25 score of 12.4 against a dense score of 0.31 makes `alpha` meaningless.

### What this library uses

`HybridRetriever` fetches `fetch_k = max(top_k * 3, 20)` candidates from each retriever, min-max normalizes both to [0, 1] independently, then fuses:

```
score = alpha * dense_norm + (1 - alpha) * bm25_norm
```

`alpha=1.0` is pure dense, `alpha=0.0` is pure BM25, `alpha=0.6` is the empirically tuned default. The 3x over-fetch ensures the normalizer sees enough candidates to produce stable min/max estimates.

### Why not RRF

Reciprocal Rank Fusion (Cormack et al., 2009) is the dominant approach in production RAG stacks (Elasticsearch, Weaviate, LangChain, LlamaIndex). Instead of scores, RRF uses rank positions:

```
RRF_score(d) = sum_r  1 / (k + rank_r(d))
```

RRF requires no normalization, is robust to outlier scores, and has no tunable alpha. The reason linear interpolation was chosen here: the alpha-sweep experiment in downstream projects is the goal. RRF has no direct equivalent to alpha. Linear interpolation makes each alpha value a controlled experiment with an interpretable semantic weight.

### Known limitations of min-max normalization

**Outlier sensitivity:** If one chunk has an unusually high BM25 score, it becomes the 1.0 ceiling and compresses all other BM25 scores toward zero. The effective alpha shifts toward dense even at `alpha=0.6`.

**Pool-size sensitivity:** Min and max come from `fetch_k` candidates, not the full corpus. A query matching many documents and one matching few produce incomparable normalized scores at the same alpha.

**Production recommendation:** Use RRF for production. Use linear interpolation when retriever weight tuning or a controlled ablation is the explicit goal. If using linear interpolation in production, replace min-max with z-score normalization to reduce outlier sensitivity.

---

## UUID Chunk IDs

`Chunk.id` is a UUID4 string. When two chunking configurations each chunk the same PDF, both produce chunk index 0 through N. Sequential IDs would collide in a shared index. UUIDs guarantee uniqueness across configs without coordination.

Tradeoff: UUIDs are opaque. Provenance is in `Chunk.source_file` and `Chunk.chunk_index`.

---

## NLTK punkt with Regex Fallback

`SentenceBasedChunker` uses NLTK `punkt` for sentence tokenization. NLTK downloads the punkt model automatically on first use (requires internet). If unavailable, the chunker falls back to a regex splitter (`r'(?<=[.!?])\s+'`). The regex handles the 95% case but breaks on abbreviations, ellipses, and URLs. The fallback is logged; it does not fail silently.

---

## Precision@K Ceiling for Single-Ground-Truth Queries

When each query has exactly one relevant chunk, `Precision@K` is capped at `1/K`. At K=5 the maximum is 0.20. This is correct TREC/BEIR behavior. MRR and Recall@K are the appropriate primary metrics for this regime. Precision@K is included for completeness.

---

## Per-Config QA Generation

`rag-pipeline-systematic-evals` generates a separate QA dataset for each chunking configuration. A shared QA set evaluated against all configs would give an unfair advantage to the config whose chunk boundaries matched the QA generation config. Per-config generation is fair but means QA datasets are not directly comparable across configs; only the IR metrics derived from them are.
