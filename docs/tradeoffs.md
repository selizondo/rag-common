# Design Decisions and Tradeoffs

## Protocol-based adapters over abstract base classes

In Python there are two ways to define a shared interface that multiple classes must implement. An **ABC (abstract base class)** requires each implementing class to explicitly inherit from it and call `super().__init__()`. A **structural protocol** (`typing.Protocol`) instead says: "any class that has these methods qualifies, regardless of inheritance." This is called duck typing — if it walks like a duck, it is a duck.

`VectorStoreProtocol` and `RetrieverProtocol` use structural protocols (`@runtime_checkable`). The benefit: adding a Qdrant or Pinecone backend is a new file with zero changes to retrieval code. The tradeoff: protocol violations are caught at type-check time (mypy), not at instantiation. A class that forgets `save()` won't fail until `save()` is called.

## IndexFlatIP with L2-normalisation (cosine via inner product)

**FAISS** (Facebook AI Similarity Search) is a library for finding the most similar vectors in a large collection quickly. Every chunk is converted to a vector (embedding) at ingest time, and at query time the query is also converted to a vector; FAISS finds the stored vectors closest to the query vector.

"Closest" can be measured in different ways. **Cosine similarity** measures the angle between two vectors: 1.0 means identical direction, 0 means unrelated, -1 means opposite. It is the standard measure for semantic similarity because it is scale-invariant — a long document and a short document on the same topic produce vectors pointing the same direction even if their magnitudes differ.

FAISS's `IndexFlatIP` computes inner products (dot products). After **L2-normalising** all embeddings at ingest time (scaling each vector to unit length), inner product equals cosine similarity. Scores land in `[-1, 1]`, which matches `HybridRetriever`'s expectation for the dense side of score fusion. The alternative — `IndexFlatL2` — returns squared Euclidean distances in `[0, 4]`, which would need a different normalisation formula in `HybridRetriever`. `IndexFlatIP` is the standard choice for semantic search with normalised embeddings.

Scale boundary: `IndexFlatIP` performs **exhaustive O(N) search** — it compares the query against every stored vector, one by one. For small corpora this is fast enough; at ~1M chunks it becomes too slow for interactive use. The upgrade path is `IndexIVFFlat` (approximate nearest-neighbour search, which trades a small accuracy loss for much faster lookup) or a hosted **ANN** (Approximate Nearest Neighbour) service like Pinecone or Weaviate. This boundary is not yet reached in either `rag-pipeline-systematic-evals` or `rag-pipeline-experimentation`.

## Score fusion strategy in HybridRetriever

### Why combine two retrievers at all?

There are two fundamentally different ways to find relevant chunks for a query.

**Keyword retrieval (BM25)** counts how often the query's words appear in each chunk, weighted by how rare those words are across the corpus. It is fast and precise for exact-match lookups ("what is the half-life of carbon-14?") but fails completely when the query and the answer use different words ("radioactive decay duration" vs "half-life").

**Semantic / dense retrieval** converts both the query and every chunk into high-dimensional vectors (embeddings) using a neural model trained to place similar meanings near each other. It handles paraphrase and synonymy well but can retrieve topically adjacent chunks that don't actually answer the question.

Hybrid retrieval combines both signals: BM25 catches precise keyword matches that dense retrieval would miss; dense retrieval catches semantic matches that BM25 would miss. The challenge is that their scores are on completely different scales.

### The problem: two incompatible score spaces

**BM25** scores are unbounded positive floats, computed by a TF-IDF-style formula that rewards term frequency and penalises common words. A query with rare technical terms can produce a score of 12 or higher for the top result; a query with common words might top out at 1.5. There is no fixed ceiling.

**Dense** scores are cosine similarities in `[-1, 1]`. Because embeddings are L2-normalised, two vectors can only ever be at most 1.0 apart in the cosine sense. Real-world top scores for semantic search typically fall in the range `[0.2, 0.9]`.

Fusing raw scores directly lets BM25 dominate — a high-frequency keyword query might produce a BM25 score of 12.4 against a dense score of 0.31, making `alpha` meaningless.

### What this library uses: linear interpolation with min-max normalisation

**Min-max normalisation** rescales each set of scores onto the same `[0, 1]` range before combining them. The formula is:

```
normalised = (score - min_score) / (max_score - min_score)
```

The lowest-scoring candidate in the pool becomes 0.0; the highest becomes 1.0; everything else scales proportionally. After normalisation, both BM25 and dense scores live on the same scale and can be meaningfully added together.

The combined score is then:

```
score = alpha * dense_norm + (1 - alpha) * bm25_norm
```

`HybridRetriever` fetches `fetch_k = max(top_k * 3, 20)` candidates from each retriever, normalises both sets independently, then fuses. The 3× over-fetch ensures the normaliser sees enough candidates to produce stable min/max estimates — if you only fetch 1 candidate, min equals max and normalisation is undefined.

`alpha` is a first-class tunable: `alpha=1.0` is pure dense, `alpha=0.0` is pure BM25, `alpha=0.6` weights dense at 60% and BM25 at 40%. This makes the notebook's alpha-sweep experiment meaningful — each value is a genuine operating point with an interpretable semantic weight.

### The industry standard: Reciprocal Rank Fusion (RRF)

The dominant approach in production RAG stacks (Elasticsearch, Weaviate, LangChain, LlamaIndex) is RRF (Cormack et al., 2009). Instead of working with raw scores at all, RRF converts each retriever's output into a ranked list (1st, 2nd, 3rd…) and combines those positions:

```
RRF_score(d) = Σ_r  1 / (k + rank_r(d))
```

Intuitively: a document that ranks 1st in both retrievers gets the highest fused score; one that ranks 20th in both gets a low score. The actual numeric scores from BM25 or the vector store are discarded entirely — only the ordering matters. This is why no normalisation is needed: rank positions 1, 2, 3 are already on a universal scale regardless of which retriever produced them.

`k=60` is a smoothing constant that prevents a rank-1 result from dominating too heavily (without it, `1/1 = 1.0` dwarfs `1/2 = 0.5`). The value 60 was established empirically by Cormack et al. as robust across many IR benchmarks.

Key properties:

- **No normalisation required.** Rank positions are already on a common ordinal scale across all retrievers.
- **Robust to outlier scores.** A BM25 score of 50 vs. 0.3 becomes rank 1 vs. rank 2 — the magnitude difference disappears.
- **No tunable alpha.** Both retrievers contribute equally by default. Weighted RRF exists (`w_r / (k + rank_r)`) but is rarely tuned in practice.
- **Pool-independent.** Adding or removing candidates doesn't re-scale existing scores, unlike min-max.

### Why linear interpolation was chosen here

| Property | Linear (this lib) | RRF |
|---|---|---|
| Tunable retriever weight | Yes (`alpha`) | No (or requires separate `w_r`) |
| Score stability across query batches | Pool-dependent | Stable |
| Outlier sensitivity | High (min-max distorted by extreme scores) | None |
| Interpretability of alpha sweep | Direct — each alpha is a controlled experiment | N/A |
| Implementation complexity | Low | Low |

The alpha sweep in `rag_common_tour.ipynb` is the deciding factor. The goal is a controlled experiment showing how retrieval quality shifts across the dense-to-sparse spectrum. RRF has no direct equivalent to alpha, so it cannot produce that experiment. Linear interpolation is the right design for this use case.

### Known limitations of min-max normalisation

**Pool contamination (outlier sensitivity).** Min-max rescales every score relative to the highest and lowest in the pool. If one chunk has an unusually high BM25 score — say, because it quotes the query terms many times — it becomes the new "1.0" ceiling and all other BM25 scores are compressed toward zero. The dense scores are unaffected, so `alpha` no longer has its intended weight: you asked for 60% dense but get something closer to 90% dense in practice. The 3× over-fetch reduces the chance that one outlier is the only high scorer, but does not eliminate the problem.

**Pool-size sensitivity.** The min and max values that define the normalisation range come only from the `fetch_k` candidates retrieved by each retriever, not from the full corpus. With 5 total chunks, the pool contains everything and normalisation is stable. With 10,000 chunks, the pool contains only the top 20 BM25 hits, and their min/max reflects those 20 documents, not the full distribution. A query that matches many documents and a query that matches few will produce incomparable normalised scores even at the same `alpha`.

**Score meaning changes across queries.** A fused score of 0.7 on one query does not mean the same thing as 0.7 on another, because both min and max are query-specific. This makes it unreliable to use the raw fused score as an absolute quality threshold (e.g. "only return results above 0.6"). **MRR** (Mean Reciprocal Rank — how highly the first correct answer is ranked on average) and **Recall@K** (what fraction of relevant documents appear in the top K results) are rank-based and therefore better evaluation metrics for this retriever than raw score comparisons.

### Production recommendation

Use RRF as the default fusion strategy for production systems. Switch to linear interpolation only when: (a) retriever weight tuning is an explicit product requirement, or (b) a controlled ablation study is the goal. In either case, replace min-max with **z-score normalisation** (subtract the mean, divide by the standard deviation) to reduce outlier sensitivity — z-scoring is less distorted by a single extreme value because it uses the mean and spread of all scores rather than just the min and max. Evaluate on a held-out query set rather than tuning alpha on the retrieval corpus, since alpha tuned on the same data it will retrieve from will overfit to that corpus's score distribution.

## UUID chunk IDs (not sequential integers)

A **UUID** (Universally Unique Identifier) is a 128-bit randomly generated string like `c2838a97-e494-4d95-8ad8-581738f101ef`. The probability of two independently generated UUIDs colliding is astronomically low — effectively zero for any practical corpus size.

`Chunk.id` is a UUID4 (randomly generated UUID) string. When two grid-search configs each chunk the same PDF, config A produces chunk index 0 through N; config B also produces 0 through N. Sequential IDs would collide in any shared index or result set — chunk 3 from config A and chunk 3 from config B are different chunks but would have the same ID. UUIDs guarantee uniqueness across configs without coordination. The tradeoff: UUIDs are opaque — you can't tell from the ID which document or chunk position it came from. Provenance is in `Chunk.source_file` and `Chunk.chunk_index`.

## NLTK punkt with regex fallback

Splitting text into sentences is harder than it looks. A naive split on `.` breaks on "Dr. Smith", "e.g.", URLs, and decimal numbers. **NLTK's punkt tokeniser** is a pre-trained model that handles these cases by learning which periods end sentences and which don't from a large text corpus.

`SentenceBasedChunker` uses NLTK `punkt` for sentence tokenisation. NLTK downloads the punkt model automatically on first use (requires internet on first run). If NLTK is unavailable (import error or model missing), the chunker falls back to a **regex splitter** (`r'(?<=[.!?])\s+'`) — a simple pattern that splits after any sentence-ending punctuation followed by whitespace. The regex handles the 95% case but breaks on abbreviations ("Dr. Smith"), ellipses, and URLs. The fallback is documented and logged; it does not fail silently.

## Precision@K ceiling with single ground-truth chunks

**Precision@K** answers: "Of the K results I returned, what fraction were relevant?" If there is only one correct chunk for a query and you retrieve K=5 results, the best you can ever score is 1/5 = 0.20 — because at most one of the five results can be the correct one, and the other four slots are necessarily "wrong" by definition.

When each query has exactly one relevant chunk (as in `rag-pipeline-systematic-evals`'s synthetic QA), `Precision@K` is therefore capped at `1/K` regardless of retriever quality. At K=5 the maximum achievable score is 0.20. This is expected behaviour, not a bug. **MRR** (did the single correct answer appear near the top?) and **Recall@K** (did the single correct answer appear anywhere in the top K?) are the appropriate metrics for this regime. `Precision@K` is included in the output for completeness and to enable comparisons across different K values.

## Per-config QA generation (not shared ground truth)

`rag-pipeline-systematic-evals` generates a separate QA dataset for each chunking configuration using the chunks produced by that config. The alternative — one shared QA set evaluated against all configs — would mean the ground truth was created from one chunk boundary set and evaluated against another, introducing an unfair advantage for the config whose chunks match the QA generation config. Per-config generation is fair but means the QA datasets are not directly comparable across configs, only the IR metrics derived from them are.
