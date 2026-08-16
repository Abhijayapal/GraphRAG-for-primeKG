"""
backend/embeddings/pykeen_searcher.py

Wrapper around a trained PyKEEN RotatE model (.pkl file).

This replaces the naive cosine-similarity approach (EmbeddingSearcher)
with the correct RotatE triple-scoring function:

    score(drug, relation, disease) = -|| h ∘ r - t ||

WHY this is better than cosine similarity:
    RotatE entity embeddings live in drug-space and disease-space
    respectively. A drug vector and a disease vector may look
    geometrically far apart even if the drug treats the disease — they
    are NOT meant to be compared directly.

    The relation embedding r acts as a learned "rotation" that maps
    drug-space into disease-space. Using score_hrt() applies that
    rotation, giving a biologically meaningful similarity score.
"""

from __future__ import annotations

import csv
import io
import logging
import pickle
from typing import Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Relations used for drug repurposing prediction
REPURPOSE_RELATIONS = ["indication", "off-label use"]


class PyKEENSearcher:
    """
    Loads a PyKEEN RotatE model from a .pkl file and provides
    triple-based drug-disease scoring.

    Args:
        model_path:    Path to trained_model.pkl
        train_csv:     Path to train.csv (used to rebuild entity/relation maps)
        test_csv:      Path to test.csv  (used to add any extra entities)
    """

    def __init__(
        self,
        model_path: str,
        train_csv: str = "train.csv",
        test_csv: str = "test.csv",
    ) -> None:
        self._model_path = model_path
        self._train_csv  = train_csv
        self._test_csv   = test_csv
        self._model      = None
        self._entity_to_id:   dict[str, int] = {}
        self._relation_to_id: dict[str, int] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def load(self) -> None:
        """Load model and rebuild entity/relation maps from CSV."""
        logger.info("Loading PyKEEN RotatE model from %s", self._model_path)

        class _CPUUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch.storage" and name == "_load_from_bytes":
                    return lambda b: torch.load(
                        io.BytesIO(b), map_location="cpu", weights_only=False
                    )
                return super().find_class(module, name)

        with open(self._model_path, "rb") as f:
            self._model = _CPUUnpickler(f).load()
        self._model.eval()

        logger.info(
            "Model loaded: %d entities, dim=%d, dtype=%s",
            self._model.num_entities,
            self._model.entity_representations[0](indices=None).shape[-1],
            self._model.entity_representations[0](indices=None).dtype,
        )

        self._build_maps()
        logger.info(
            "Entity map: %d | Relation map: %d",
            len(self._entity_to_id),
            len(self._relation_to_id),
        )

    def score_repurposing(self, drug_name: str, disease_name: str) -> float:
        """
        Score how well a drug could repurpose for a disease using the
        RotatE triple scoring function averaged over INDICATION relations.

        Returns:
            float in range [0, 1] (normalised logistic of raw score)
        """
        drug_idx    = self._entity_to_id.get(drug_name)
        disease_idx = self._entity_to_id.get(disease_name)
        if drug_idx is None or disease_idx is None:
            return 0.0

        scores = []
        for rel_name in REPURPOSE_RELATIONS:
            rel_idx = self._relation_to_id.get(rel_name)
            if rel_idx is None:
                continue
            raw = self._score_hrt(drug_idx, rel_idx, disease_idx)
            scores.append(raw)

        if not scores:
            return 0.0

        raw_avg = float(sum(scores) / len(scores))
        # Normalise to [0, 1] via sigmoid
        return float(torch.sigmoid(torch.tensor(raw_avg)).item())

    def rank_drugs_for_disease(
        self,
        disease_name: str,
        candidate_drugs: list[str],
        top_k: int = 20,
    ) -> list[dict]:
        """
        Rank a list of candidate drug names for a disease using triple scoring.

        Returns:
            List of {drug, rotate_score} sorted by score desc.
        """
        disease_idx = self._entity_to_id.get(disease_name)
        if disease_idx is None:
            logger.warning("Disease '%s' not in entity map", disease_name)
            return []

        rel_indices = [
            self._relation_to_id[r]
            for r in REPURPOSE_RELATIONS
            if r in self._relation_to_id
        ]
        if not rel_indices:
            return []

        results = []
        for drug_name in candidate_drugs:
            drug_idx = self._entity_to_id.get(drug_name)
            if drug_idx is None:
                continue
            scores = [self._score_hrt(drug_idx, r, disease_idx) for r in rel_indices]
            raw = float(sum(scores) / len(scores))
            norm = float(torch.sigmoid(torch.tensor(raw)).item())
            results.append({"name": drug_name, "rotate_score": norm})

        results.sort(key=lambda x: -x["rotate_score"])
        return results[:top_k]

    def has_entity(self, name: str) -> bool:
        return name in self._entity_to_id

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _score_hrt(self, head_idx: int, rel_idx: int, tail_idx: int) -> float:
        """Run RotatE scoring function for a single triple.
        Input format: tensor of shape [1, 3] = [[head, rel, tail]]
        """
        hrt = torch.tensor([[head_idx, rel_idx, tail_idx]], dtype=torch.long)
        with torch.no_grad():
            score = self._model.score_hrt(hrt)
        return float(score.item())

    def _build_maps(self) -> None:
        """Load entity_to_id and relation_to_id from exported JSON maps."""
        import json
        import os
        
        # Determine the directory of the model path
        model_dir = os.path.dirname(self._model_path)
        if not model_dir:
            model_dir = "embeddings/rotate_data"
            
        entity_map_path = os.path.join(model_dir, "rotate_entity_map.json")
        relation_map_path = os.path.join(model_dir, "rotate_relation_map.json")
        
        try:
            with open(entity_map_path, "r", encoding="utf-8") as f:
                self._entity_to_id = json.load(f)
            with open(relation_map_path, "r", encoding="utf-8") as f:
                self._relation_to_id = json.load(f)
        except FileNotFoundError as e:
            logger.error("Map JSON not found! Make sure to run dump_relation_map.py. Error: %s", e)
            raise
