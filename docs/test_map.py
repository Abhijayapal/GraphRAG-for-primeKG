import json
from benchmark_recall import build_maps

e, r, d = build_maps()

with open("embeddings/rotate_data/rotate_entity_map.json") as f:
    true_map = json.load(f)

# Compare the first few items
print("My map length:", len(e))
print("True map length:", len(true_map))

mismatches = 0
for k, v in true_map.items():
    if k not in e:
        print(f"Key {k} not in my map!")
        mismatches += 1
        continue
    if e[k] != v:
        if mismatches < 5:
            print(f"Mismatch for {k}: true={v}, mine={e[k]}")
        mismatches += 1

print("Total mismatches:", mismatches)
