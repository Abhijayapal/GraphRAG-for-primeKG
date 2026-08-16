"""
benchmark_recall.py
-------------------
Measures Recall@10 of the hybrid retrieval pipeline against ground-truth
drug-disease indication pairs from PrimeKG.

Business Interpretation:
    Recall@10 = "What % of the time does the correct known drug appear
                 in the top-10 results when we query its disease?"

Target: >80% Recall@10 for known indication pairs.

Usage:
    python benchmark_recall.py

Output:
    Recall@1  : 0.XX
    Recall@5  : 0.XX
    Recall@10 : 0.XX  ← Primary metric
    Latency   : P50=XXms  P95=XXms  P99=XXms
"""

from __future__ import annotations

import csv
import json
import logging
import time
import os
import pickle
import io
import random
import numpy as np
import torch
from collections import defaultdict
from typing import Optional

# Silence noisy HybridRanker guardrail warnings during benchmarking
logging.basicConfig(level=logging.ERROR)

# ── CPU unpickler for model loaded on non-GPU machine ────────────────────────
class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu",
                                        weights_only=False)
        return super().find_class(module, name)


# ── Load ground-truth indication pairs from train.csv ───────────────────────
def load_ground_truth(path: str = "train.csv",
                      relations: tuple = ("indication", "off-label use"),
                      sample_size: int = 500) -> list[tuple[str, str]]:
    """Return list of (disease_name, drug_name) known positive pairs."""
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel  = row["display_relation"].strip()
            x_t  = row["x_type"].strip()
            y_t  = row["y_type"].strip()
            if rel in relations:
                if x_t == "drug" and y_t == "disease":
                    pairs.append((row["y_name"].strip(), row["x_name"].strip()))
                elif x_t == "disease" and y_t == "drug":
                    pairs.append((row["x_name"].strip(), row["y_name"].strip()))
    random.seed(42)
    return random.sample(pairs, min(sample_size, len(pairs)))


# ── Load entity/relation maps (MUST USE JSON exported from training) ───────────
def build_maps():
    import json
    # Use the actual maps exported from TriplesFactory to ensure perfect ID sync
    with open("embeddings/rotate_data/rotate_entity_map.json", "r") as f:
        entity_to_id = json.load(f)
    
    with open("embeddings/rotate_data/rotate_relation_map.json", "r") as f:
        relation_to_id = json.load(f)
        
    # We still need the list of drug names to rank them
    drug_names_set = set()
    with open("train.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["x_type"].strip() == "drug": drug_names_set.add(row["x_name"].strip())
            if row["y_type"].strip() == "drug": drug_names_set.add(row["y_name"].strip())
            
    drug_names = [d for d in drug_names_set if d in entity_to_id]
    return entity_to_id, relation_to_id, drug_names


# ── Hybrid Recall@K evaluator ─────────────────────────────────────────────────
def evaluate_hybrid_recall(pairs: list[tuple[str, str]], ks: tuple = (1, 5, 10)) -> dict:
    from backend.retrieval.hybrid_ranker import HybridRanker
    from backend.embeddings.pykeen_searcher import PyKEENSearcher
    from backend.retrieval.cypher_retriever import CypherRetriever
    from backend.graph.driver import create_driver

    print("Initializing components...")
    ranker = HybridRanker()
    
    # Init RotatE
    searcher = PyKEENSearcher(model_path="trained_model.pkl")
    searcher.load()

    # ── Pre-build drug candidate list (filter entity map to drugs only) ────────
    # This is the KEY optimization: instead of scoring ALL ~100k entities,
    # we only score the ~10k known drug entities — ~10x speedup.
    print("Building drug candidate list from train.csv...")
    drug_names_set: set[str] = set()
    with open("train.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["x_type"].strip() == "drug":
                drug_names_set.add(row["x_name"].strip())
            if row["y_type"].strip() == "drug":
                drug_names_set.add(row["y_name"].strip())
    # Only keep drugs that exist in the RotatE entity map
    drug_candidates = [d for d in drug_names_set if searcher.has_entity(d)]
    print(f"Drug candidates in RotatE map: {len(drug_candidates)} / {len(drug_names_set)} from CSV")

    # Init Neo4j Cypher Retriever
    driver = create_driver()
    cypher = CypherRetriever(driver)
    
    hits = {k: 0 for k in ks}
    valid = 0
    latencies: list[float] = []

    for idx, (disease_name, true_drug) in enumerate(pairs, 1):
        print(f"  [{idx}/{len(pairs)}] {disease_name[:50]!r}...", flush=True)
        valid += 1
        t0 = time.perf_counter()
        
        # 1. Cypher retrieval (graph-based, fast)
        c_res = cypher.retrieve(disease_name, mode="disease", top_k=50)
        
        # 2. RotatE retrieval — drugs only (NOT all entities)
        r_res = searcher.rank_drugs_for_disease(disease_name, drug_candidates, top_k=50)
        
        # 3. Hybrid RRF fusion
        fused = ranker.rank(disease_name, c_res, r_res, top_k=max(ks))
        
        latencies.append((time.perf_counter() - t0) * 1000)

        ranked_drugs = [r["name"] for r in fused["candidates"]]
        
        for k in ks:
            if true_drug in ranked_drugs[:k]:
                hits[k] += 1

    driver.close()

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.50)] if n else 0
    p95 = latencies[int(n * 0.95)] if n else 0
    p99 = latencies[int(n * 0.99)] if n else 0

    recall = {k: round(hits[k] / valid, 4) if valid else 0 for k in ks}
    return {
        "recall":   recall,
        "latency":  {"P50_ms": round(p50,1), "P95_ms": round(p95,1), "P99_ms": round(p99,1)},
        "valid_pairs": valid,
        "total_pairs": len(pairs),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading ground-truth pairs (50 samples for speed)...")
    pairs = load_ground_truth(sample_size=50)

    print(f"Evaluating Recall@K on {len(pairs)} disease-drug pairs...\n")
    results = evaluate_hybrid_recall(pairs, ks=(1, 5, 10))

    print("=" * 55)
    print("  RETRIEVAL BENCHMARK — Hybrid Ranker (Cypher + RotatE)")
    print("=" * 55)
    print(f"  Valid pairs evaluated : {results['valid_pairs']}")
    for k, v in results["recall"].items():
        flag = " <Primary metric" if k == 10 else ""
        print(f"  Recall@{k:<3}            : {v:.4f}  ({v*100:.1f}%){flag}")
    lat = results["latency"]
    print(f"  Query latency         : P50={lat['P50_ms']}ms  "
          f"P95={lat['P95_ms']}ms  P99={lat['P99_ms']}ms")
    print("=" * 55)

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to benchmark_results.json")
