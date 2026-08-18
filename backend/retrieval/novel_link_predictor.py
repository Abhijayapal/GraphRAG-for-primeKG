"""
NovelLinkPredictor: Drug Repurposing Engine.

Core biological insight:
    PrimeKG contains data up to 2019. COVID-19 came in 2020. If we can predict
    "Heparin is a good candidate for Coronavinae infectious disease" using ONLY
    2019 data, then validate it against 2020 clinical literature — that is real
    scientific value. This module does exactly that.

Ensemble Formula (Updated: proper RotatE triple scoring):
    rotate_score  = PyKEEN score_hrt(drug, indication, disease)  [0..1]
    fastrp_score  = Neo4j FastRP vector index cosine similarity  [0..1]
    ensemble      = 0.6 * rotate_score + 0.4 * fastrp_score
    bio_bonus     = +0.15 if Drug -> Gene -> Disease path exists
    final         = min(1.0, ensemble + bio_bonus)

WHY triple scoring instead of cosine similarity:
    RotatE entity embeddings live in different geometric spaces for drugs
    and diseases. Direct cosine similarity Drug vs Disease is meaningless.
    The learned relation rotation (e.g. 'indication') bridges the two spaces,
    giving a biologically accurate score via score_hrt().
"""

from __future__ import annotations

import os
import numpy as np
from dotenv import load_dotenv
from neo4j import Driver

load_dotenv()

# Relationship types that define an EXISTING drug-disease connection
EXISTING_EDGE_RELS = "INDICATION|CONTRAINDICATION|OFF_LABEL_USE"

# Relationship types that define a Drug -> Gene link (mechanistic bridge)
DRUG_GENE_RELS = "TARGET|ENZYME|TRANSPORTER|CARRIER"

# Ensemble weights
W_ROTATE  = 0.6    # PyKEEN RotatE triple score: relation-aware, highest biological signal
W_FASTRP  = 0.4    # FastRP: graph topology
BIO_BONUS = 0.15   # Bonus for mechanistic Drug -> Gene -> Disease path


class NovelLinkPredictor:
    """
    Predicts novel (currently unknown) drug-disease links for drug repurposing.

    Usage:
        predictor = NovelLinkPredictor(driver, pykeen_searcher)
        predictions = predictor.predict("Coronavinae infectious disease", top_k=20)
    """

    def __init__(self, driver: Driver, rotate_searcher):
        """
        Args:
            driver:           Neo4j driver (for FastRP + graph queries).
            rotate_searcher:  A loaded PyKEENSearcher instance (for triple scores)
                              OR legacy EmbeddingSearcher (fallback cosine).
        """
        self._driver = driver
        self._rotate = rotate_searcher
        # Detect whether we have the new PyKEEN searcher
        self._use_triple_scoring = hasattr(rotate_searcher, 'score_repurposing')

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def predict(self, disease_name: str, top_k: int = 20) -> dict:
        """
        Predict novel drug candidates for the given disease.

        Steps:
          1. Verify disease exists in the graph with FastRP embedding.
          2. Fetch all Drug names from Neo4j.
          3. Fetch drugs ALREADY connected to this disease (to exclude them).
          4. Score remaining drugs via RotatE + FastRP ensemble.
          5. Check biological plausibility (mechanistic path) for top candidates.
          6. Return ranked novel predictions.

        Returns:
            {
                "disease":      str,
                "predictions":  [
                    {
                        "drug":           str,
                        "final_score":    float,
                        "rotate_score":   float,
                        "fastrp_score":   float,
                        "bio_plausible":  bool,
                        "bio_path":       str | None,  — e.g. "Heparin -> TARGET -> F2 -> ASSOCIATED_WITH -> COVID"
                    },
                    ...
                ],
                "excluded_known": int,   — how many drugs were already connected
                "total_scored":   int,   — how many novel drugs were scored
            }
        """
        print(f"\n[NovelLinkPredictor] Disease: '{disease_name}'")

        # Step 1: Try FastRP embedding (requires Neo4j GDS — not available on free Aura)
        disease_fastrp = self._get_fastrp_embedding(disease_name)
        use_fastrp = disease_fastrp is not None
        if not use_fastrp:
            print(f"  [info] No FastRP embedding for '{disease_name}'. "
                  "Falling back to RotatE-only scoring.")

        # Step 2: Check disease is in RotatE entity map
        try:
            _ = self._rotate.get_embedding(disease_name)
            disease_in_rotate = True
        except (KeyError, RuntimeError):
            disease_in_rotate = False
            print(f"  [warn] '{disease_name}' not in RotatE entity map. "
                  "RotatE scores will be 0 for all candidates.")

        if not use_fastrp and not disease_in_rotate:
            raise ValueError(
                f"Disease '{disease_name}' not found in knowledge graph embeddings."
            )

        # Step 3: Fetch all drug names from Neo4j
        all_drugs = self._get_all_drug_names()
        print(f"  [step 3] Total drugs in graph: {len(all_drugs)}")

        # Step 4: Fetch drugs already connected to this disease
        known_drugs = self._get_known_connected_drugs(disease_name)
        print(f"  [step 4] Already connected drugs: {len(known_drugs)}")

        # Step 5: Novel candidates = all drugs minus already-connected
        novel_drugs = [d for d in all_drugs if d not in known_drugs]
        print(f"  [step 5] Novel candidate drugs: {len(novel_drugs)}")

        # Step 6: Get candidates via FastRP vector index OR RotatE cosine fallback
        if use_fastrp:
            print(f"  [step 6] Scoring top candidates via FastRP vector index...")
            fastrp_candidates = self._batch_fastrp_scores(disease_fastrp, top_k=100)
            # Filter to only novel drugs (exclude already-known)
            fastrp_candidates = [
                c for c in fastrp_candidates if c["name"] not in known_drugs
            ]
        else:
            # RotatE-only fallback: score all novel drugs by cosine similarity
            print(f"  [step 6] FastRP unavailable — scoring via RotatE cosine fallback...")
            fastrp_candidates = [
                {"name": drug, "score": 0.0} for drug in novel_drugs
            ]

        # Step 7: Score RotatE for each candidate
        # When using FastRP fallback, score ALL novel drugs via RotatE and pick top-K
        candidates_to_score = fastrp_candidates if use_fastrp else fastrp_candidates
        print(f"  [step 7] Adding RotatE scores for {len(candidates_to_score)} candidates...")
        scored = []
        for item in candidates_to_score:
            drug_name    = item["name"]
            fastrp_score = item["score"]

            # Use proper triple scoring if PyKEENSearcher is available
            if self._use_triple_scoring:
                rotate_score = self._rotate.score_repurposing(drug_name, disease_name)
            else:
                # Legacy fallback: cosine similarity
                rotate_score = self._rotate_cosine(drug_name, disease_name)

            # When no FastRP, weight is entirely on RotatE
            if use_fastrp:
                ensemble = W_ROTATE * rotate_score + W_FASTRP * fastrp_score
            else:
                ensemble = rotate_score

            scored.append({
                "drug":          drug_name,
                "rotate_score":  round(rotate_score, 4),
                "fastrp_score":  round(fastrp_score, 4),
                "ensemble":      round(ensemble, 4),
                "bio_plausible": False,
                "bio_path":      None,
            })

        # Sort by ensemble score, then check bio plausibility for top-K only
        scored.sort(key=lambda x: -x["ensemble"])
        top_candidates = scored[:top_k]

        # Step 8: Biological plausibility check for top-K
        # WHY only top-K: each plausibility check = 1 Neo4j query.
        # Checking all 6000 drugs would take too long.
        print(f"  [step 8] Checking biological plausibility for top {len(top_candidates)} candidates...")
        for item in top_candidates:
            path = self._find_bio_path(item["drug"], disease_name)
            if path:
                item["bio_plausible"] = True
                item["bio_path"] = path
                item["final_score"] = min(1.0, item["ensemble"] + BIO_BONUS)
            else:
                item["final_score"] = item["ensemble"]

        # Final sort by final_score
        top_candidates.sort(key=lambda x: (-x["final_score"], x["drug"].lower()))

        print(f"  [done] Top prediction: {top_candidates[0]['drug']} "
              f"(score={top_candidates[0]['final_score']:.4f})")

        return {
            "disease":        disease_name,
            "predictions":    top_candidates,
            "excluded_known": len(known_drugs),
            "total_scored":   len(novel_drugs),
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _get_all_drug_names(self) -> set:
        """Fetch all Drug node names from Neo4j."""
        with self._driver.session() as session:
            rows = session.run("MATCH (d:Drug) RETURN d.name AS name").data()
        return {r["name"] for r in rows if r["name"]}

    def _get_known_connected_drugs(self, disease_name: str) -> set:
        """
        Fetch all drugs that already have ANY edge to this disease.
        WHY "any" edge and not just INDICATION:
            We want ALL known connections — CONTRAINDICATION, OFF_LABEL_USE included.
            We only predict where truly no relationship exists.
        """
        with self._driver.session() as session:
            rows = session.run(
                f"""
                MATCH (d:Drug)-[:{EXISTING_EDGE_RELS}]->(dis:Disease {{name: $disease}})
                RETURN d.name AS name
                """,
                disease=disease_name,
            ).data()
        return {r["name"] for r in rows if r["name"]}

    def _get_fastrp_embedding(self, disease_name: str) -> list | None:
        """Fetch the FastRP embedding vector for the disease node."""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (dis:Disease {name: $name}) RETURN dis.fastrp_embedding AS emb",
                name=disease_name,
            ).single()
        if result and result["emb"]:
            return list(result["emb"])
        return None

    def _batch_fastrp_scores(self, disease_embedding: list, top_k: int) -> list:
        """
        Batch-query the Neo4j FastRP vector index to find the top-K most similar
        Drug nodes to the given disease embedding.

        WHY batch instead of per-drug: querying 6000+ drugs one-by-one would take
        minutes. Vector index search is O(log n) and returns all top-K in one query.
        """
        with self._driver.session() as session:
            hits = session.run(
                """
                CALL db.index.vector.queryNodes('fastrp_index', $k, $vec)
                YIELD node, score
                WHERE node:Drug
                RETURN node.name AS name, score
                LIMIT $limit
                """,
                k=top_k * 3,
                vec=disease_embedding,
                limit=top_k,
            ).data()
        return [{"name": h["name"], "score": h["score"]} for h in hits]

    def _rotate_cosine(self, drug_name: str, disease_name: str) -> float:
        """
        Compute cosine similarity between drug and disease RotatE embeddings.

        WHY cosine and not dot product:
            Raw RotatE embeddings have varying magnitudes. Cosine similarity
            normalizes for magnitude, giving a pure directional similarity.
            This is the same normalization we do in FAISS (L2 normalize then inner product).
        """
        try:
            drug_emb    = self._rotate.get_embedding(drug_name)
            disease_emb = self._rotate.get_embedding(disease_name)
        except (KeyError, RuntimeError):
            return 0.0

        # Cosine similarity = dot(a,b) / (|a| * |b|)
        dot    = np.dot(drug_emb, disease_emb)
        norm_a = np.linalg.norm(drug_emb)
        norm_b = np.linalg.norm(disease_emb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _find_bio_path(self, drug_name: str, disease_name: str) -> str | None:
        """
        Check if a mechanistic bridge exists: Drug -> Gene -> Disease.
        This is the biological plausibility check.

        WHY this matters:
            A drug that has HIGH embedding similarity to a disease could be a
            statistical artefact. But if we ALSO find a real gene-level pathway
            connecting them, the prediction has mechanistic grounding — much
            stronger scientific evidence.

        Returns a human-readable path string if found, None otherwise.
        """
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (d:Drug {{name: $drug}})
                      -[r1:{DRUG_GENE_RELS}]->(g:Gene)
                      -[r2:ASSOCIATED_WITH]->(dis:Disease {{name: $disease}})
                RETURN d.name AS drug, type(r1) AS rel1,
                       g.name AS gene, type(r2) AS rel2,
                       dis.name AS disease
                LIMIT 1
                """,
                drug=drug_name,
                disease=disease_name,
            ).single()

        if result:
            return (
                f"{result['drug']} "
                f"--[{result['rel1']}]--> "
                f"{result['gene']} "
                f"--[{result['rel2']}]--> "
                f"{result['disease']}"
            )
        return None
