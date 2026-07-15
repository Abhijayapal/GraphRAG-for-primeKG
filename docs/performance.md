# Performance

## Benchmark Setup

- Machine: Intel i7, 16 GB RAM, SSD
- Neo4j Desktop (local), no GPU
- Graph: PrimeKG (~120K nodes, ~8M edges)
- Embeddings: 22,741 entity vectors, 128 dimensions

---

## API Latency

| Operation | p50 | p95 |
|---|---|---|
| Entity resolution (Cypher) | 15 ms | 35 ms |
| Cypher graph traversal (50 paths) | 90 ms | 200 ms |
| FAISS ANN search | < 2 ms | < 5 ms |
| RRF fusion (Python, in-memory) | < 1 ms | < 1 ms |
| **POST /query (end-to-end)** | **~110 ms** | **~240 ms** |
| **POST /repurpose** | **~1.8 s** | **~3 s** |
| Application startup | ~3 s | — |

**Bottleneck:** Neo4j Cypher traversal dominates `/query` latency. FAISS is negligible.

**`/repurpose` is slower** because it runs two Cypher queries (all drugs, existing edges) and a plausibility check for each top-K candidate before returning.

---

## Memory

| Resource | Size |
|---|---|
| RotatE embedding matrix (.npy) | ~46 MB |
| FAISS flat index | ~22 MB |
| Neo4j page cache (configured) | 2 GB |
| Python process (API only) | ~250 MB |

The FAISS index and embedding matrix are kept in RAM for the lifetime of the process. This trades memory for sub-millisecond search latency.

---

## Startup Time

| Step | Time |
|---|---|
| Neo4j driver connect + verify | ~0.5 s |
| Load .npy embedding matrix | ~0.3 s |
| L2-normalise embeddings | ~0.2 s |
| Build FAISS index | ~2.0 s |
| Instantiate retrieval components | ~0.1 s |
| **Total** | **~3.1 s** |

All of this happens once in the `lifespan` context manager before the server accepts any requests.

---

## Retrieval Quality

Bootstrap edge-recovery experiment (100 random edges deleted, system asked to recover them):

| Metric | Value |
|---|---|
| Average Hit@20 | 47% |
| Best single run | 80% |
| Random baseline (20 / ~6,600 drugs) | 0.3% |
| Improvement over random | ~157× |

---

## Scalability Notes

**Current constraints:**
- Single-process uvicorn — vertical scaling only
- FAISS index is in-memory, not shared across processes
- Neo4j connection pool (50) handles concurrent requests up to that limit

**Incremental improvements that add real value:**
- LRU cache on entity resolution (same canonical names are queried repeatedly)
- LRU cache on frequent Cypher queries (popular diseases: Alzheimer, Parkinson)
- Increase Neo4j page cache to fit more of the graph in RAM

**Larger changes (out of scope for this project):**
- Multiple uvicorn workers with a shared FAISS index (requires IPC or a vector DB)
- GraphSAGE inductive embeddings (handles new nodes without retraining)
