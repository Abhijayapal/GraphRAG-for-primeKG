import csv
import os
import time
from collections import defaultdict
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI       = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER      = os.getenv("NEO4J_USER", "neo4j")
PASSWORD  = os.getenv("NEO4J_PASSWORD")
CSV_FILE  = "kg_filtered.csv"
BATCH_SIZE = 500   # for nodes (they are simple MERGEs)
EDGE_BATCH  = 100  # smaller for edges (MATCH+MATCH+MERGE is heavier)

TYPE_TO_LABEL = {
    "drug":               "Drug",
    "disease":            "Disease",
    "gene/protein":       "Gene",
}

def get_label(raw: str) -> str:
    return TYPE_TO_LABEL.get(raw.strip().lower(), "Entity")

def to_rel_type(display: str) -> str:
    return display.strip().upper().replace(" ", "_").replace("/", "_").replace("-", "_")

# 

def create_constraints(session):
    print("Creating constraints...")
    stmts = [
        "CREATE CONSTRAINT drug_id       IF NOT EXISTS FOR (n:Drug)               REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT disease_id    IF NOT EXISTS FOR (n:Disease)            REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT gene_id       IF NOT EXISTS FOR (n:Gene)               REQUIRE n.id IS UNIQUE",
    ]
    for s in stmts:
        try:
            session.run(s)
        except Exception as e:
            print(f"  (skip) {e}")
    print("  Done.\n")

# 

def read_csv():
    print(f"Reading {CSV_FILE} into memory...")
    nodes_by_label = defaultdict(dict)   # label -> {id: {id, name, source}}
    edges_by_type  = defaultdict(list)   # rel_type -> [{x_id, y_id}]

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            xl = get_label(row["x_type"])
            yl = get_label(row["y_type"])
            rt = to_rel_type(row["display_relation"])

            nodes_by_label[xl][row["x_id"]] = {"id": row["x_id"], "name": row["x_name"], "source": row["x_source"]}
            nodes_by_label[yl][row["y_id"]] = {"id": row["y_id"], "name": row["y_name"], "source": row["y_source"]}
            edges_by_type[rt].append({"x_id": row["x_id"], "y_id": row["y_id"], "xl": xl, "yl": yl})

            if i % 50000 == 0:
                print(f"  Read {i:,} rows...")

    total_nodes = sum(len(v) for v in nodes_by_label.values())
    total_edges = sum(len(v) for v in edges_by_type.values())
    print(f"  Unique nodes: {total_nodes:,} across {len(nodes_by_label)} labels")
    print(f"  Edges:        {total_edges:,} across {len(edges_by_type)} relationship types\n")
    return nodes_by_label, edges_by_type

#load nodes 

def load_nodes(session, nodes_by_label):
    print("Loading nodes...")
    t0 = time.time()
    total = 0

    for label, nodes_dict in nodes_by_label.items():
        rows = list(nodes_dict.values())
        print(f"  [{label}] {len(rows):,} nodes", end="", flush=True)

        # Split into batches
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            session.run(
                f"""
                UNWIND $rows AS row
                MERGE (n:{label} {{id: row.id}})
                ON CREATE SET n.name = row.name, n.source = row.source
                """,
                rows=batch
            )
            total += len(batch)

        print(f"  OK")

    print(f"  Nodes done: {total:,} in {time.time()-t0:.1f}s\n")

# ─── Step 3: load edges ───────────────────────────────────────────────────────

def load_edges(session, edges_by_type):
    print("Loading edges...")
    t0 = time.time()
    total = 0

    for rel_type, rows in edges_by_type.items():
        count = len(rows)
        print(f"  [{rel_type}] {count:,} edges", end="", flush=True)

        # Group by (x_label, y_label) pair so MATCH uses the indexed constraint
        from collections import defaultdict as dd
        by_labels = dd(list)
        for r in rows:
            by_labels[(r["xl"], r["yl"])].append(r)

        for (xl, yl), sub_rows in by_labels.items():
            for i in range(0, len(sub_rows), EDGE_BATCH):
                batch = sub_rows[i:i + EDGE_BATCH]
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (x:{xl} {{id: row.x_id}})
                    MATCH (y:{yl} {{id: row.y_id}})
                    MERGE (x)-[:{rel_type}]->(y)
                    """,
                    rows=batch
                )
                total += len(batch)
                if total % 5000 == 0:
                    print(".", end="", flush=True)

        print(f"  OK")

    print(f"  Edges done: {total:,} in {time.time()-t0:.1f}s\n")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print(f"Connecting to {URI}...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    nodes_by_label, edges_by_type = read_csv()

    with driver.session() as session:
        create_constraints(session)
        load_nodes(session, nodes_by_label)
        load_edges(session, edges_by_type)

    driver.close()
    print(f"\nTotal time: {time.time()-t_start:.1f}s")

if __name__ == "__main__":
    main()
