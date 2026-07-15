# Design Trade-offs

Decisions made during implementation, with the alternatives that were considered and rejected.

---

## Why Neo4j over a relational database?

**Problem:** Drug-disease connections often exist through intermediate nodes — proteins, pathways, genes. A query like "find all drugs connected to Alzheimer disease through a shared gene target" requires traversing 3 hops.

In a relational database this would be:
```sql
SELECT d.name FROM drugs d
JOIN drug_gene dg ON d.id = dg.drug_id
JOIN genes g ON dg.gene_id = g.id
JOIN gene_disease gd ON g.id = gd.gene_id
WHERE gd.disease_id = 'alzheimer_id'
```

Three JOINs across large tables — slow and hard to reason about. Adding a fourth hop multiplies the complexity.

In Neo4j, the same query is:
```cypher
MATCH (drug:Drug)-[:TARGETS]->(gene:Gene)-[:ASSOCIATED_WITH]->(disease:Disease)
WHERE disease.name = "Alzheimer disease"
RETURN drug
```

**Decision:** Neo4j. Graph traversal is the primary access pattern. The GDS plugin also provides FastRP and Node2Vec directly on the in-memory graph projection.

---

## Why Hybrid retrieval instead of graph-only or vector-only?

**The problem with graph-only (Cypher):**
- High precision — results have explicit biological evidence
- Low recall — misses drugs that are similar in embedding space but have no direct or 1-hop path

**The problem with vector-only (FAISS):**
- Higher recall — captures latent similarity
- Lower precision — no path to explain why the result was returned
- Cannot distinguish relationship types (TREATS vs SIDE_EFFECT)

**Hybrid (Cypher + RotatE, fused with RRF):**
- Recall improves because both retrieval methods contribute candidates
- Precision is maintained because Cypher results are weighted higher (0.6 vs 0.4)
- Every result carries a source attribution — recruiters/clinicians can see whether the evidence is graph-based, embedding-based, or both

In drug discovery, missing a valid candidate (false negative) is worse than including a noisy one (false positive). Hybrid retrieval optimises for recall without completely sacrificing precision.

---

## Why Reciprocal Rank Fusion over weighted score averaging?

Cypher scores (rule-based, 0.3–1.2) and RotatE cosine similarities (0.75–0.99) are on incompatible scales. Directly averaging them gives Cypher scores disproportionate weight at higher values.

RRF uses only rank position:
```
rrf(rank) = 1 / (60 + rank)
```

This is scale-invariant — a drug ranked #1 by Cypher contributes the same amount as a drug ranked #1 by RotatE, regardless of their raw score values.

**Alternative considered:** Min-max normalisation then weighted average. Rejected because normalisation amplifies outlier noise and requires knowing the score distribution in advance, which varies per query.

---

## Why RotatE over Node2Vec for graph embeddings?

PrimeKG has 30+ relationship types: TREATS, BINDS_TO, ASSOCIATED_WITH, SIDE_EFFECT, OFF_LABEL_USE, etc.

**Node2Vec** uses biased random walks. All edge types are treated identically — the walk does not distinguish whether it crossed a TREATS edge or a SIDE_EFFECT edge.

**RotatE** models each relation type as a rotation in complex vector space. The geometric transformation for TREATS is fundamentally different from BINDS_TO. Drugs and diseases are positioned in embedding space relative to the specific relation types connecting them.

For a heterogeneous biomedical graph where relation type carries most of the signal, RotatE is the appropriate choice. Node2Vec and FastRP are kept as baselines for the benchmarking framework.

---

## Why FastAPI over Flask?

| Requirement | Flask | FastAPI |
|---|---|---|
| Request body validation | Manual (marshmallow/pydantic separately) | Built-in Pydantic models |
| OpenAPI docs | Requires flask-restx or flasgger | Auto-generated at `/docs` |
| Async lifespan | Requires workarounds | Native `lifespan` context manager |
| Type hints | Optional | First-class |
| Startup hook (modern) | `@app.before_first_request` (deprecated) | `lifespan` (stable, recommended) |

The lifespan pattern is the main reason. Loading the FAISS index (3 seconds) at startup, before the server accepts requests, requires a clean async context manager. Flask's equivalent was deprecated in 2.3.

---

## Why Pydantic BaseSettings over os.getenv()?

Scattered `os.getenv("NEO4J_URI")` calls have several problems:
- No single place to see all required configuration
- No type coercion — everything is a string
- No validation — a missing variable fails silently mid-request
- No documentation of what each variable does

`backend/config/settings.py` declares every variable with type, default, and a description. If `NEO4J_PASSWORD` is missing, the application raises a `ValidationError` at startup with a clear message instead of failing mid-request with a generic auth error.

---

## Why one-time FAISS index loading over per-request computation?

Computing cosine similarity against 22,000+ vectors takes ~15ms naively per request. Building the FAISS index takes ~2 seconds.

The index is built once at startup and kept in RAM. Per-request search takes < 2ms. The trade-off is ~22 MB of memory held for the application lifetime — acceptable on any machine that can run the application.

**Alternative:** Rebuild the index on each request (or lazily on first request). Rejected because the first user would experience a 2+ second delay, and concurrent first-requests would each trigger a rebuild race.

---

## Batch graph loading over single-row inserts

Loading 8M edges one-by-one via individual Cypher queries would take hours and generate 8M network round-trips to Neo4j.

`scripts/load_kg.py` uses batched `UNWIND` transactions:
```cypher
UNWIND $batch AS row
MERGE (a:Drug {id: row.source_id})
MERGE (b:Disease {id: row.target_id})
MERGE (a)-[:TREATS]->(b)
```

500 rows per transaction reduces round-trips from 8M to ~16,000. Batch size of 500 balances per-transaction memory against round-trip overhead — larger batches risk transaction timeouts on constrained systems.
