"""
RotateTrainer: Trains a RotatE Knowledge Graph Embedding model using PyKEEN
              and saves the resulting entity embeddings to disk.

Workflow:
    1. Load the .tsv triples file into a PyKEEN TriplesFactory.
    2. Split into 80% train / 20% test (saved for Week 5 evaluation).
    3. Train the RotatE model (128-dim, 100 epochs).
    4. Extract entity embedding vectors from the trained model.
    5. Save embeddings as rotate_embeddings.npy  (numpy array).
    6. Save name→index map as rotate_entity_map.json.

Output files (in embeddings/rotate_data/):
    rotate_embeddings.npy   — shape [num_entities, 128], float32
    rotate_entity_map.json  — {"Metformin": 4521, "schizophrenia": 89, ...}
    test_triples.tsv        — held-out edges for Week 5 benchmarking
"""

import json
import os

import numpy as np
import torch
#  and paste C:\ml\Scripts\python.exe 
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


# Configuration

EMBEDDING_DIM = 128          # Must match FastRP/Node2Vec for fair benchmarking
NUM_EPOCHS    = 100          # Standard for biomedical KGs (balance quality/speed)
TRAIN_RATIO   = 0.8          # 80% train, 20% test (standard ML split)
RANDOM_SEED   = 42           # For reproducibility


class RotateTrainer:

    def __init__(self, triples_path: str, output_dir: str):
        """
        Args:
            triples_path: Path to the .tsv file produced by TripleExporter.
            output_dir:   Directory to save embeddings and entity map.
        """
        self.triples_path = triples_path
        self.output_dir   = output_dir

    # Public API
    def train(self):
        """
        Full pipeline: load → split → train → extract → save.
        This is the main function to call.
        """
        print("=== RotatE Training Pipeline ===")
        os.makedirs(self.output_dir, exist_ok=True)

        # Step 1: Load triples into TriplesFactory
        tf = self._load_triples()

        # Step 2: Split into train/test
        train_tf, test_tf = self._split(tf)

        # Step 3: Train RotatE model
        model = self._train_model(train_tf, test_tf)

        # Step 4 & 5: Extract and save embeddings
        self._save_embeddings(model, tf)

        # Step 6: Save test triples for Week 5 evaluation
        self._save_test_triples(test_tf)

        print("=== Training Complete ===")

    # Private helpers

    def _load_triples(self) -> TriplesFactory:
        """
        Load the .tsv file into a PyKEEN TriplesFactory.

        TriplesFactory automatically:
          - Assigns a unique integer ID to every entity name (Drug, Gene, Disease).
          - Assigns a unique integer ID to every relationship type.
          - Builds an internal lookup table (entity_to_id, relation_to_id).
        This integer mapping is required because neural networks work with
        numbers, not strings.
        """
        print(f"  [load] Loading triples from '{self.triples_path}'...")
        tf = TriplesFactory.from_path(
            self.triples_path,
            create_inverse_triples=True,  # Add reverse edges (improves RotatE quality)
        )
        print(f"  [load] {tf.num_triples} triples, "
              f"{tf.num_entities} entities, "
              f"{tf.num_relations} relation types.")
        return tf

    def _split(self, tf: TriplesFactory):
        """
        Split triples into 80% training and 20% test sets.

        Why split?
            We hide 20% of edges from the model during training.
            In Week 5, we check whether the model can predict those hidden
            edges — this is the standard Link Prediction evaluation protocol.
        """
        print(f"  [split] Splitting: {int(TRAIN_RATIO*100)}% train / "
              f"{int((1-TRAIN_RATIO)*100)}% test...")

        train_tf, test_tf = tf.split(
            ratios=TRAIN_RATIO,
            random_state=RANDOM_SEED,
        )
        print(f"  [split] Train: {train_tf.num_triples} | "
              f"Test: {test_tf.num_triples}")
        return train_tf, test_tf

    def _train_model(self, train_tf: TriplesFactory, test_tf: TriplesFactory):
        """
        Train the RotatE model using PyKEEN's high-level pipeline() function.

        Key settings explained:
          model='RotatE'     → The model architecture (rotation in complex space).
          embedding_dim=128  → Each entity gets a 128-float vector. Must match
                               FastRP/Node2Vec dim for fair ablation comparison.
          num_epochs=100     → The model sees all training triples 100 times.
                               More epochs = better quality, but slower.
          loss='NSSALoss'    → "Negative Sampling Self-Adversarial Loss".
                               Industry standard for KGE link prediction tasks.
          optimizer='Adam'   → Standard neural network optimizer.
          random_seed=42     → Reproducibility (same results every run).
        """
        print(f"  [train] Training RotatE ({EMBEDDING_DIM}-dim, {NUM_EPOCHS} epochs)...")
        print("  [train] This may take 30-60 minutes on CPU.")

        result = pipeline(
            training=train_tf,
            testing=test_tf,
            model="RotatE",
            model_kwargs={"embedding_dim": EMBEDDING_DIM},
            training_kwargs={"num_epochs": NUM_EPOCHS},
            optimizer="Adam",
            loss="NSSALoss",
            random_seed=RANDOM_SEED,
            device="cpu",
        )

        print("  [train] Training complete!")
        return result.model

    def _save_embeddings(self, model, tf: TriplesFactory):
        """
        Extract the final entity embedding matrix from the trained RotatE model
        and save it to disk as two files:

        1. rotate_embeddings.npy
           A 2D numpy array of shape [num_entities, 128].
           Row i contains the 128-float embedding vector for entity i.

        2. rotate_entity_map.json
           A dictionary mapping entity name → row index.
           e.g. {"Metformin": 4521, "schizophrenia": 89, ...}

        Why two files?
           When searching later, we need to:
             a) Look up the row index for "schizophrenia" (using entity_map).
             b) Grab row 89 from the numpy array (using embeddings.npy).
           Without the map, we wouldn't know which row belongs to which entity.
        """
        print("  [save] Extracting entity embeddings from model...")

        # Get the embedding matrix as a numpy array
        # model.entity_representations[0] → the entity embedding table
        # .detach() → stop tracking gradients (we're done training)
        # .cpu()    → move from GPU memory to CPU if necessary
        # .numpy()  → convert PyTorch tensor to numpy array
        entity_embeddings = (
            model.entity_representations[0](indices=None)
            .detach()
            .cpu()
            .numpy()
        )

        # Save the numpy array to disk
        embeddings_path = os.path.join(self.output_dir, "rotate_embeddings.npy")
        np.save(embeddings_path, entity_embeddings)
        print(f"  [save] Embeddings saved: {entity_embeddings.shape} to '{embeddings_path}'")

        # Build and save the name → index map
        # tf.entity_to_id is a dict like {"Metformin": 4521, ...}
        entity_map_path = os.path.join(self.output_dir, "rotate_entity_map.json")
        with open(entity_map_path, "w", encoding="utf-8") as f:
            json.dump(tf.entity_to_id, f, ensure_ascii=False, indent=2)
        print(f"  [save] Entity map saved to '{entity_map_path}'")

    def _save_test_triples(self, test_tf: TriplesFactory):
        """
        Save the held-out test triples to a TSV file for Week 5 benchmarking.
        We convert integer IDs back to entity/relation names before saving
        so the evaluation script can read them as human-readable strings.
        """
        print("  [save] Saving test triples for Week 5 evaluation...")

        test_path = os.path.join(self.output_dir, "test_triples.tsv")

        # id_to_entity and id_to_relation are reverse lookup dicts
        id_to_entity   = {v: k for k, v in test_tf.entity_to_id.items()}
        id_to_relation = {v: k for k, v in test_tf.relation_to_id.items()}

        with open(test_path, "w", encoding="utf-8") as f:
            for triple in test_tf.mapped_triples.numpy():
                head = id_to_entity[triple[0]]
                rel  = id_to_relation[triple[1]]
                tail = id_to_entity[triple[2]]
                f.write(f"{head}\t{rel}\t{tail}\n")

        print(f"  [save] Test triples saved to '{test_path}'")
