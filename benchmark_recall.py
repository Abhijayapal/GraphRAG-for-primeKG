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
import time
import os
import pickle
import io
import random
import numpy as np
import torch
from collections import defaultdict
from typing import Optional

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


# ── Build entity/relation maps (must match model training) ───────────────────
def build_maps(train_path: str = "train.csv"):
    entities, relations = set(), set()
    drug_names_set = set()
    with open(train_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            x, y = row["x_name"].strip(), row["y_name"].strip()
            entities.add(x); entities.add(y)
            relations.add(row["display_relation"].strip())
            if row["x_type"].strip() == "drug": drug_names_set.add(x)
            if row["y_type"].strip() == "drug": drug_names_set.add(y)
    entity_to_id  = {n: i for i, n in enumerate(sorted(entities))}
    relation_to_id = {r: i for i, r in enumerate(sorted(relations))}
    drug_names = [d for d in drug_names_set if d in entity_to_id]
    return entity_to_id, relation_to_id, drug_names


# ── Recall@K evaluator ────────────────────────────────────────────────────────
def evaluate_recall(model, entity_to_id, relation_to_id, drug_names,
                    pairs: list[tuple[str, str]],
                    ks: tuple = (1, 5, 10)) -> dict:
    """
    For each (disease, true_drug) pair:
      1. Score all drugs via RotatE indication relation
      2. Check if true_drug appears in top-K
    """
    drug_indices = [entity_to_id[d] for d in drug_names]
    drug_tensor  = torch.tensor(drug_indices)

    # Get relation IDs for indication / off-label use
    rel_ids = [
        relation_to_id[r]
        for r in ("indication", "off-label use")
        if r in relation_to_id
    ]

    hits = {k: 0 for k in ks}
    valid = 0
    latencies: list[float] = []

    for disease_name, true_drug in pairs:
        if disease_name not in entity_to_id or true_drug not in entity_to_id:
            continue
        if entity_to_id[true_drug] not in drug_indices:
            continue

        disease_idx = entity_to_id[disease_name]
        valid += 1

        # Aggregate scores across indication relations
        t0 = time.perf_counter()
        combined = torch.zeros(len(drug_indices))
        for rel_id in rel_ids:
            h = drug_tensor.unsqueeze(1)
            r = torch.full((len(drug_indices), 1), rel_id)
            t = torch.full((len(drug_indices), 1), disease_idx)
            hrt = torch.cat([h, r, t], dim=1)
            with torch.no_grad():
                combined += model.score_hrt(hrt).squeeze(-1)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

        # Rank drugs (higher score = better)
        sorted_idx = combined.argsort(descending=True).tolist()
        ranked_drugs = [drug_names[i] for i in sorted_idx]

        for k in ks:
            if true_drug in ranked_drugs[:k]:
                hits[k] += 1

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
    print("Loading model...")
    with open("trained_model.pkl", "rb") as f:
        model = CPUUnpickler(f).load()
    model.eval()

    print("Building entity maps...")
    entity_to_id, relation_to_id, drug_names = build_maps()

    print("Loading ground-truth pairs (500 samples)...")
    pairs = load_ground_truth(sample_size=500)

    print(f"Evaluating Recall@K on {len(pairs)} disease-drug pairs...\n")
    results = evaluate_recall(model, entity_to_id, relation_to_id,
                              drug_names, pairs, ks=(1, 5, 10))

    print("=" * 55)
    print("  RETRIEVAL BENCHMARK — Knowledge Graph Embeddings (RotatE)")
    print("=" * 55)
    print(f"  Valid pairs evaluated : {results['valid_pairs']}")
    for k, v in results["recall"].items():
        flag = " ← Primary metric" if k == 10 else ""
        print(f"  Recall@{k:<3}            : {v:.4f}  ({v*100:.1f}%){flag}")
    lat = results["latency"]
    print(f"  Query latency         : P50={lat['P50_ms']}ms  "
          f"P95={lat['P95_ms']}ms  P99={lat['P99_ms']}ms")
    print("=" * 55)

    # Save as JSON for README/report
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to benchmark_results.json")
