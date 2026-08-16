from backend.embeddings.pykeen_searcher import PyKEENSearcher
import time

s = PyKEENSearcher(model_path="trained_model.pkl", train_csv="train.csv", test_csv="train.csv")
s.load()

# Test one known pair
drug = "Heparin" # Or some other drug
disease = "Coronavirinae infectious disease"
print("Score for Heparin -> COVID:", s.score_repurposing(drug, disease))

# Let's pick a known indication from train.csv
import csv
known_pair = None
with open("train.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["display_relation"] == "indication":
            if row["y_type"] == "drug" and row["x_type"] == "disease":
                known_pair = (row["x_name"], row["y_name"])
                break
            elif row["x_type"] == "drug" and row["y_type"] == "disease":
                known_pair = (row["y_name"], row["x_name"])
                break

if known_pair:
    disease, drug = known_pair
    print(f"Known pair: {disease} treated by {drug}")
    score = s.score_repurposing(drug, disease)
    print("Score:", score)
    
    t0 = time.time()
    # Let's rank all drugs for this disease
    drugs = [k for k in s._entity_to_id.keys()] # Just a large subset
    ranked = s.rank_drugs_for_disease(disease, drugs, top_k=50)
    print("Ranked in", time.time() - t0, "seconds")
    print("Top 5:", ranked[:5])
    
    # Check rank of true drug
    for i, r in enumerate(ranked):
        if r["drug"] == drug:
            print(f"True drug {drug} found at rank {i+1}")
            break
