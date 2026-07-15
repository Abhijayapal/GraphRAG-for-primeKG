"""
TripleExporter: Reads every edge from Neo4j and saves them as a TSV file
                of triplets (head, relation, tail) for PyKEEN.

Why export to a file?
    PyKEEN does not connect to databases directly. It reads plain text files.
    A TSV file is universal — it works with any KGE library, not just PyKEEN.
    Once exported, re-training doesn't require Neo4j to be running.

Output file format (tab-separated, no header):
    Metformin\tINDICATION\tType 2 Diabetes
    DRD2\tASSOCIATED_WITH\tSchizophrenia
    ...
"""

import os
from neo4j import Driver


# All relationship types we want to include in training
# We include every type so RotatE learns the geometry of all 8 relationships
EXPORT_REL_TYPES = [
    "INDICATION",
    "CONTRAINDICATION",
    "OFF_LABEL_USE",
    "TARGET",
    "ENZYME",
    "TRANSPORTER",
    "CARRIER",
    "ASSOCIATED_WITH",
]


class TripleExporter:

    def __init__(self, driver: Driver, output_path: str):
        """
        Args:
            driver:      Live Neo4j driver (database must be running).
            output_path: Path to save the .tsv file (e.g. 'data/triples.tsv').
        """
        self.driver = driver
        self.output_path = output_path

    # Public API

    def export(self) -> int:
        """
        Export all edges from Neo4j as (head, relation, tail) triplets.

        Returns:
            Total number of triplets exported.
        """
        

        # Make sure the output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        total = 0
        with open(self.output_path, "w", encoding="utf-8") as f:
            with self.driver.session() as session:
                for rel_type in EXPORT_REL_TYPES:
                    count = self._export_rel_type(session, f, rel_type)
                    print(f"Done")
                    total += count

        print(f"Done")
        return total

    # Private helpers

    def _export_rel_type(self, session, file_handle, rel_type: str) -> int:
        """
        Query all edges of a given type and write them to the file.
        Each line is: head_name <TAB> relation_type <TAB> tail_name

        We use .name property for all nodes because it is the canonical
        human-readable identifier that maps back to our EntityResolver.
        """
        # Dynamic Cypher: fetch all edges of this specific relationship type
        rows = session.run(
            f"""
            MATCH (h)-[r:{rel_type}]->(t)
            WHERE h.name IS NOT NULL AND t.name IS NOT NULL
            RETURN h.name AS head, type(r) AS relation, t.name AS tail
            """
        ).data()

        count = 0
        for row in rows:
            # Write one triplet per line, tab-separated
            file_handle.write(f"{row['head']}\t{row['relation']}\t{row['tail']}\n")
            count += 1

        return count
