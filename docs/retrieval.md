# Retrieval Pipeline

## Entity Resolution

Before any retrieval, raw user input is mapped to a canonical graph node name.

**Step 1 — Alias table** (no database round-trip):
Common names that do not appear in the graph under their modern name are resolved deterministically. For example, "covid" → "Coronavinae infectious disease" because the graph uses 2019-era terminology.

**Step 2 — Cypher fuzzy match** (if alias table misses):
Three strategies are tried in priority order:
1. Exact (`toLower(n.name) = $q`) — score 1.0
2. Prefix (`STARTS WITH`) — score 0.7
3. Contains — score 0.4

Results are deduplicated across node labels and sorted by score. The top match is used as the canonical name for all downstream retrieval.

---

## Cypher Retrieval

`CypherRetriever` runs two query patterns and collects all results:

**Direct paths:**
```cypher
MATCH (drug:Drug)-[r]->(disease:Disease)
WHERE disease.name = $name
RETURN drug, type(r) AS rel_type, "direct" AS path_type
```

**1-hop via gene:**
```cypher
MATCH (drug:Drug)-[r1]->(gene:Gene)-[r2]->(disease:Disease)
WHERE disease.name = $name
RETURN drug, gene, type(r1) AS rel_type, "1-hop" AS path_type
```

Results are passed to `CypherRanker` which assigns a base score per `(rel_type, path_type)` combination:

| Edge type | Path | Base score |
|---|---|---|
| INDICATION | direct | 1.0 |
| OFF_LABEL_USE | direct | 0.6 |
| TARGET | 1-hop | 0.4 |
| ENZYME / TRANSPORTER / CARRIER | 1-hop | 0.3 |
| Unknown | any | 0.1 |

A multi-evidence bonus (+0.2) is added when the same drug appears in both a direct path and a 1-hop path — two independent biological lines of evidence.

---

## Embedding Search

`EmbeddingSearcher` wraps a FAISS flat index built over RotatE entity embeddings. At startup:

1. Load embedding matrix from `.npy` file
2. L2-normalise all vectors (converts inner product to cosine similarity)
3. Build a `IndexFlatIP` FAISS index
4. Add all vectors

At query time, the entity name is looked up in the entity map, its vector is retrieved, normalised, and searched against the index. FAISS returns the top-K nearest neighbours by cosine similarity.

If an entity is not in the embedding index (e.g. a new node added after training), the search falls back gracefully to an empty result list — Cypher results are still used.

---

## Hybrid Ranking (RRF)

The two ranked lists (Cypher and RotatE) are fused using Reciprocal Rank Fusion:

```
rrf(rank) = 1 / (K + rank),   K = 60
```

Final score:
```
hybrid_score = 0.6 × rrf(cypher_rank) + 0.4 × rrf(rotate_rank)
```

If an entity appears in only one list, the other term contributes 0. A drug ranked #1 in both lists scores higher than one ranked #1 in only one list.

K=60 is the standard value from Cormack et al. (2009). It dampens the outsized advantage of rank-1 over rank-2 while still rewarding high-ranked results.

---

## Novel Link Prediction

`NovelLinkPredictor` handles `POST /repurpose`. Unlike `/query`, it specifically targets entity pairs with **no existing edge**:

1. Retrieve all Drug nodes from Neo4j (~6,600)
2. Filter out any drug that already has any relationship to the target disease
3. For remaining candidates, compute RotatE cosine similarity
4. For top-K candidates, run a Cypher check for `Drug→Gene→Disease` paths (biological plausibility)
5. Final score = RotatE similarity + 0.2 × (1 if plausible path exists)
6. Return sorted by final score

This is the link prediction mode — not retrieval of known facts, but inference of missing connections from learned embedding geometry.
