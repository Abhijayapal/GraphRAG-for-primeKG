# Architecture

## Components

| Module | Responsibility |
|---|---|
| `backend/api/main.py` | App factory, lifespan startup, router registration |
| `backend/api/dependencies.py` | Injects shared state into route handlers via `Depends()` |
| `backend/api/routes/` | One file per endpoint group (health, query, repurpose, explain) |
| `backend/config/settings.py` | Single source of truth for all environment variables |
| `backend/graph/driver.py` | Creates and validates the Neo4j driver with connection pooling |
| `backend/retrieval/entity_resolver.py` | Maps raw input to canonical graph node names |
| `backend/retrieval/cypher_retriever.py` | Executes parameterised Cypher path queries |
| `backend/retrieval/cypher_ranker.py` | Scores Cypher results by edge type and path depth |
| `backend/retrieval/hybrid_ranker.py` | Fuses Cypher and embedding lists with RRF |
| `backend/retrieval/novel_link_predictor.py` | Scores entity pairs with no existing graph edge |
| `backend/embeddings/embedding_searcher.py` | Builds and queries the FAISS ANN index |
| `backend/utils/logger.py` | Configures structured logging once at startup |

---

## Startup Sequence

```
python -m uvicorn backend.api.main:app
          │
          ▼
   lifespan() context manager
          │
          ├── 1. Create Neo4j driver (verify connectivity)
          ├── 2. Load RotatE embedding matrix (.npy)
          ├── 3. L2-normalise embeddings
          ├── 4. Build FAISS flat index
          ├── 5. Instantiate retrieval components
          │      EntityResolver, CypherRetriever, CypherRanker,
          │      HybridRanker, NovelLinkPredictor
          └── 6. Attach all objects to app.state
                 → server accepts requests
```

All heavy work happens once. Subsequent requests read from `app.state` — no re-initialisation.

---

## Request Lifecycle — POST /query

```
POST /query  {"name": "alzheimer", "entity_type": "disease", "top_k": 10}
     │
     ▼
 FastAPI route handler (routes/query.py)
     │
     ├── EntityResolver.resolve("alzheimer", entity_type="Disease")
     │      ├── check alias table → "Alzheimer disease"
     │      └── if no alias: Cypher exact/prefix/contains match
     │
     ├── CypherRetriever.retrieve("Alzheimer disease", mode="disease")
     │      ├── direct Drug→Disease edges (Cypher)
     │      └── Drug→Gene→Disease 1-hop paths (Cypher)
     │
     ├── CypherRanker.rank(cypher_results)
     │      ├── score each result by edge type
     │      └── apply multi-evidence bonus
     │
     ├── EmbeddingSearcher.search("Alzheimer disease", top_k=50)
     │      └── FAISS inner-product search over normalised vectors
     │
     └── HybridRanker.rank(cypher_ranked, rotate_results, top_k=10)
            └── RRF → sorted candidates with source attribution
     │
     ▼
 JSON response with candidates + latency_ms
```

---

## Dependency Injection Pattern

Route handlers do not access `app.state` directly. Instead, they declare typed dependencies:

```python
# routes/query.py
def query(
    req: QueryRequest,
    resolver: ResolverDep,      # ← injected
    retriever: RetrieverDep,    # ← injected
    h_ranker: HybridRankerDep,  # ← injected
    ...
):
```

`ResolverDep` is defined in `dependencies.py` as:

```python
def get_resolver(request: Request) -> EntityResolver:
    return request.app.state.resolver

ResolverDep = Annotated[EntityResolver, Depends(get_resolver)]
```

This makes every route's dependencies explicit and replaceable — in tests, the mock is set directly on `app.state` without modifying any route code.
