# Design Decisions and Tradeoffs

## Protocol-based adapters over abstract base classes

`VectorStoreProtocol` and `RetrieverProtocol` are structural protocols (`@runtime_checkable`) rather than ABCs. Any class that implements the required methods satisfies the protocol — no `super().__init__()`, no registration. The benefit: adding a Qdrant or Pinecone backend is a new file with zero changes to retrieval code. The tradeoff: protocol violations are caught at type-check time (mypy), not at instantiation. A class that forgets `save()` won't fail until `save()` is called.

## IndexFlatIP with L2-normalisation (cosine via inner product)

FAISS's `IndexFlatIP` computes inner products. After L2-normalising all embeddings at ingest time, inner product equals cosine similarity. Scores land in `[-1, 1]`, which matches `HybridRetriever`'s expectation for the dense side of score fusion. The alternative — `IndexFlatL2` — returns squared Euclidean distances in `[0, 4]`, which would need a different normalisation formula in `HybridRetriever`. `IndexFlatIP` is the standard choice for semantic search with normalised embeddings.

Scale boundary: `IndexFlatIP` performs exhaustive O(N) search. At ~1M chunks it becomes too slow for interactive use. The upgrade path is `IndexIVFFlat` (approximate) or a hosted ANN service. This boundary is not yet reached in either P3 or P4.

## Min-max normalisation in HybridRetriever before score fusion

BM25 scores are unbounded positive floats (TF-IDF weighted, corpus-dependent). Dense scores are cosine similarities in `[-1, 1]`. Fusing raw scores directly lets BM25 dominate for short queries with high-frequency terms. `HybridRetriever` fetches `fetch_k = max(top_k * 3, 20)` candidates from each retriever, min-max normalises both independently to `[0, 1]`, then fuses: `score = alpha * dense_norm + (1 - alpha) * bm25_norm`. The 3× over-fetch ensures the normalisation has enough candidates to produce stable min/max estimates even for queries that match few documents.

## UUID chunk IDs (not sequential integers)

`Chunk.id` is a UUID4 string. When two grid-search configs each chunk the same PDF, config A produces chunk index 0 through N; config B also produces 0 through N. Sequential IDs would collide in any shared index or result set. UUIDs guarantee uniqueness across configs without coordination. The tradeoff: UUIDs are opaque — you can't tell from the ID which document or chunk position it came from. Provenance is in `Chunk.source_file` and `Chunk.chunk_index`.

## NLTK punkt with regex fallback

`SentenceBasedChunker` uses NLTK `punkt` for sentence tokenisation. NLTK downloads the punkt corpus automatically on first use (requires internet on first run). If NLTK is unavailable (import error or corpus missing), the chunker falls back to a regex splitter (`r'(?<=[.!?])\s+'`). The regex handles the 95% case but breaks on abbreviations ("Dr. Smith"), ellipses, and URLs. The fallback is documented and logged; it does not fail silently.

## Precision@K ceiling with single ground-truth chunks

When each query has exactly one relevant chunk (as in P3's synthetic QA), `Precision@K` is capped at `1/K`. At K=5 the maximum achievable score is 0.20. This is expected behaviour, not a bug — there is only one correct chunk, so 4 of 5 retrieved results must be non-relevant. MRR and Recall@K are the primary metrics for this data regime. `Precision@K` is included in the output for completeness and to enable comparisons across different K values.

## Per-config QA generation (not shared ground truth)

P3 generates a separate QA dataset for each chunking configuration using the chunks produced by that config. The alternative — one shared QA set evaluated against all configs — would mean the ground truth was created from one chunk boundary set and evaluated against another, introducing an unfair advantage for the config whose chunks match the QA generation config. Per-config generation is fair but means the QA datasets are not directly comparable across configs, only the IR metrics derived from them are.
