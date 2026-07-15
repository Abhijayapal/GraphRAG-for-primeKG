"""
EmbeddingSearcher: Loads saved RotatE embeddings from disk and uses FAISS
                   to find the top-K most similar entities to a given query.

Why FAISS instead of sklearn cosine_similarity?
    sklearn is brute-force: it compares the query vector against every single
    vector in the array. For 100K+ entities that is slow (~seconds per query).
    FAISS builds an ANN (Approximate Nearest Neighbor) index that uses clever
    clustering/quantization to skip irrelevant regions, returning results in
    milliseconds — even for 1 million vectors.

Usage:
    searcher = EmbeddingSearcher(
        embeddings_path="embeddings/rotate_data/rotate_embeddings.npy",
        entity_map_path="embeddings/rotate_data/rotate_entity_map.json",
    )
    searcher.load()
    results = searcher.search("schizophrenia", top_k=10)
"""

import json

import faiss
import numpy as np


class EmbeddingSearcher:

    def __init__(self, embeddings_path: str, entity_map_path: str):
        """
        Args:
            embeddings_path: Path to the numpy .npy file saved by RotateTrainer.
            entity_map_path: Path to the JSON file (entity_name → row_index).
        """
        self.embeddings_path = embeddings_path
        self.entity_map_path = entity_map_path

        # These are populated by load()
        self._embeddings  = None   # numpy array: shape [num_entities, 128], normalized (for FAISS)
        self._raw_embeddings = None  # numpy array: original RotatE embeddings before normalization
        self._entity_map  = None   # dict: {"Metformin": 4521, ...}
        self._index_map   = None   # dict: {4521: "Metformin", ...} (reverse lookup)
        self._faiss_index = None   # the FAISS ANN index

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def load(self):
        """
        Load embeddings and entity map from disk, then build the FAISS index.
        Must be called before search().
        """
        print("=== EmbeddingSearcher ===")
        self._load_embeddings()
        self._load_entity_map()
        self._build_faiss_index()
        print("=== Ready for search. ===")

    def search(self, entity_name: str, top_k: int = 10) -> list:
        """
        Find the top-K most similar entities to a given entity name.

        Args:
            entity_name: The name of the entity to search from (e.g. 'schizophrenia').
            top_k:       Number of similar entities to return.

        Returns:
            List of dicts: [{"name": ..., "score": ...}, ...]
            Sorted by similarity score (highest first).

        Raises:
            KeyError: If entity_name is not found in the entity map.
        """
        if self._faiss_index is None or self._entity_map is None or self._embeddings is None or self._index_map is None:
            raise RuntimeError("Call load() before search().")

        # Step 1: Look up the row index for this entity name
        if entity_name not in self._entity_map:
            raise KeyError(
                f"Entity '{entity_name}' not found in embedding map. "
                f"Did you use the exact name from the database?"
            )
        row_idx = self._entity_map[entity_name]

        # Step 2: Grab its embedding vector from the numpy array
        # FAISS expects a 2D array even for a single query, so we reshape
        query_vector = self._embeddings[row_idx].reshape(1, -1)

        # Step 3: Search FAISS — returns distances and row indices of top-K matches
        # For a cosine index, the "distance" is actually the cosine similarity score
        # Bug 2 fix: clamp k so it never exceeds the number of indexed vectors
        k = min(top_k + 1, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(query_vector, k)
        # +1 because the top result will be the entity itself (score 1.0), which we skip

        # Step 4: Convert row indices back to entity names using the reverse map
        results = []
        for score, idx in zip(scores[0], indices[0]):
            name = self._index_map.get(int(idx))
            if name is None or name == entity_name:
                # Skip unknown indices and the query entity itself
                continue
            results.append({"name": name, "score": float(score)})

        return results[:top_k]

    def get_embedding(self, entity_name: str) -> np.ndarray:
        """
        Return the raw embedding vector for a given entity (before normalization).
        Useful for hybrid ranking in Week 4.

        Args:
            entity_name: The canonical entity name.

        Returns:
            numpy array of shape [embedding_dim] — original RotatE embedding
        """
        if self._entity_map is None or self._raw_embeddings is None:
            raise RuntimeError("Call load() before get_embedding().")
        if entity_name not in self._entity_map:
            raise KeyError(f"Entity '{entity_name}' not found in embedding map.")
        row_idx = self._entity_map[entity_name]
        # Bug 3 fix: return from _raw_embeddings, not _embeddings which is normalized in-place
        return self._raw_embeddings[row_idx]

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _load_embeddings(self):
        """
        Load the numpy array from disk.
        The array shape is [num_entities, embedding_dim] (e.g. [50000, 128]).
        Each row i is the embedding vector for the entity with id=i.

        Bug 1 fix: RotatE stores embeddings as complex numbers (torch.cfloat).
        When saved by rotate_trainer.py, the .npy file has dtype complex64.
        We must convert to float32 using np.abs() (modulus/magnitude), NOT
        .astype('float32') which silently discards the imaginary part.
        """
        print(f"  [load] Loading embeddings from '{self.embeddings_path}'...")
        raw = np.load(self.embeddings_path)

        if np.iscomplexobj(raw):
            # RotatE complex embeddings: take magnitude (|a + bi| = sqrt(a²+b²))
            # This preserves the full information encoded in both real + imaginary parts
            print("  [load] Complex embeddings detected (RotatE). Converting via np.abs()...")
            self._embeddings = np.abs(raw).astype("float32")
        else:
            self._embeddings = raw.astype("float32")

        # astype('float32') is required — FAISS only works with 32-bit floats
        print(f"  [load] Embeddings shape: {self._embeddings.shape}")

    def _load_entity_map(self):
        """
        Load the entity_name → row_index mapping from JSON.
        Also build the reverse map (row_index → entity_name) for look-up
        after FAISS returns integer indices.
        """
        print(f"  [load] Loading entity map from '{self.entity_map_path}'...")
        with open(self.entity_map_path, "r", encoding="utf-8") as f:
            self._entity_map = json.load(f)

        # Build reverse map: int(row_index) → entity_name
        self._index_map = {int(v): k for k, v in self._entity_map.items()}
        print(f"  [load] {len(self._entity_map)} entities loaded.")

    def _build_faiss_index(self):
        """
        Build a FAISS IndexFlatIP (Inner Product) index over the embeddings.

        Why Inner Product for cosine similarity?
            If we L2-normalize all vectors first (so every vector has length 1.0),
            then the Inner Product between two vectors equals their Cosine Similarity.
            This is the standard trick used in all ANN similarity search systems.

        Steps:
            1. L2-normalize all embedding vectors (in-place).
            2. Create a FAISS IndexFlatIP index (brute-force inner product).
            3. Add all normalized embedding vectors to the index.

        Note: IndexFlatIP is brute-force but exact. For very large datasets (>1M),
              switch to IndexIVFFlat for approximate but faster search.
        """
        print("  [faiss] Normalizing embeddings and building FAISS index...")

        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded. Call _load_embeddings() first.")

        # Bug 3 fix: save original embeddings BEFORE in-place normalization
        # faiss.normalize_L2 modifies self._embeddings in-place, which would corrupt
        # get_embedding() — so we keep the raw copy for external use
        self._raw_embeddings = self._embeddings.copy()

        # L2-normalize: divide each vector by its own length (in-place)
        # After this, every vector has magnitude = 1.0
        # Inner Product on unit vectors == Cosine Similarity
        faiss.normalize_L2(self._embeddings)

        # Create an Inner Product index for the embedding dimension
        embedding_dim = self._embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(embedding_dim)

        # Add all vectors to the index
        self._faiss_index.add(self._embeddings)
        print(f"  [faiss] Index built. {self._faiss_index.ntotal} vectors indexed.")
