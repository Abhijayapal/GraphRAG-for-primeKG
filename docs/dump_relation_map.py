import json
import pandas as pd
from pykeen.triples import TriplesFactory

train_path = "train.csv"
df_train = pd.read_csv(train_path)
cols = ['x_name', 'display_relation', 'y_name']
df_train = df_train[cols].dropna()

train_triples = df_train.values.astype(str)
train_factory = TriplesFactory.from_labeled_triples(
    triples=train_triples,
    create_inverse_triples=True 
)

print(train_factory.relation_to_id)

with open("embeddings/rotate_data/rotate_relation_map.json", "w") as f:
    json.dump(train_factory.relation_to_id, f, indent=2)
