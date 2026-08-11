"""
RotatE Trainer (From CSV)
-------------------------
Trains a RotatE Knowledge Graph Embedding model using PyKEEN
directly from the PrimeKG formatted train.csv and test.csv files.

It saves the resulting entity embeddings to disk with a '_v2' suffix
to ensure the old embeddings are not overwritten.
"""

import json
import os
import numpy as np
import pandas as pd
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# Configuration
EMBEDDING_DIM = 128
NUM_EPOCHS    = 100
RANDOM_SEED   = 42

def load_and_prep_data(train_path: str, test_path: str):
    print("Loading CSV files...")
    
    # Read the CSVs
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Extract only the 3 columns we need: Head, Relation, Tail
    cols = ['x_name', 'display_relation', 'y_name']
    
    # Drop any rows with NaN values just in case
    df_train = df_train[cols].dropna()
    df_test = df_test[cols].dropna()
    
    print(f"Extracted {len(df_train)} training edges and {len(df_test)} testing edges.")
    
    # Convert to numpy string arrays for PyKEEN
    train_triples = df_train.values.astype(str)
    test_triples = df_test.values.astype(str)
    
    print("Building PyKEEN Triples Factories...")
    # create_inverse_triples=True handles the reverse edge problem mathematically
    train_factory = TriplesFactory.from_labeled_triples(
        triples=train_triples,
        create_inverse_triples=True 
    )
    
    # The test factory MUST share the exact same entity/relation IDs as the train factory
    test_factory = TriplesFactory.from_labeled_triples(
        triples=test_triples,
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id
    )
    
    return train_factory, test_factory

def train_and_save(train_factory, test_factory, output_dir: str):
    print("\nStarting PyKEEN Pipeline (Model: RotatE)...")
    print(f"Entities: {train_factory.num_entities}")
    print(f"Relations: {train_factory.num_relations}")
    
    # Run the training pipeline
    result = pipeline(
        training=train_factory,
        testing=test_factory,
        model='RotatE',
        model_kwargs={'embedding_dim': EMBEDDING_DIM},
        training_kwargs={'num_epochs': NUM_EPOCHS},
        optimizer='Adam',
        random_seed=RANDOM_SEED,
        device='cpu' # Use 'gpu' if CUDA is available
    )
    
    print("\nTraining Complete! Extracting embeddings...")
    
    # Extract the trained entity embeddings matrix [num_entities, 128]
    model = result.model
    entity_embeddings_tensor = model.entity_representations[0](indices=None)
    embeddings_np = entity_embeddings_tensor.detach().cpu().numpy()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the embeddings array (using _v2 to protect old data)
    emb_path = os.path.join(output_dir, "rotate_embeddings_v2.npy")
    np.save(emb_path, embeddings_np)
    
    # Save the entity name -> row index mapping
    map_path = os.path.join(output_dir, "rotate_entity_map_v2.json")
    with open(map_path, "w") as f:
        json.dump(train_factory.entity_to_id, f, indent=2)
        
    print(f"\nSUCCESS!")
    print(f"Saved {embeddings_np.shape} embeddings to: {emb_path}")
    print(f"Saved entity map to: {map_path}")

if __name__ == "__main__":
    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"
    OUT_DIR = "embeddings/rotate_data"
    
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        print("ERROR: train.csv or test.csv not found in the root directory.")
    else:
        tr_fac, te_fac = load_and_prep_data(TRAIN_CSV, TEST_CSV)
        train_and_save(tr_fac, te_fac, OUT_DIR)
