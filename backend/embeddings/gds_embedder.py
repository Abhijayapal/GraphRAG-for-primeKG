"""
GDSEmbedder: Runs FastRP and Node2Vec inside Neo4j GDS and stores
             the resulting vectors as node properties.

Workflow:
    1. Drop any old graph projection (clean slate).
    2. Create a new in-memory projection of the full graph.
    3. Run FastRP  → writes fastrp_embedding  onto every projected node.
    4. Run Node2Vec → writes node2vec_embedding onto every projected node.
    5. Write both embeddings back to the persistent Neo4j nodes.
    6. Drop the projection to free RAM.
"""

from neo4j import Driver
GRAPH_NAME = "biomedical_graph" #in-memory GDS graph projection
NODE_LABELS = ["Drug", "Gene", "Disease"] #nodes to be included in projection 
# Which relationship types to include in the projection
REL_TYPES = [
    "INDICATION",
    "CONTRAINDICATION",
    "OFF_LABEL_USE",
    "TARGET",
    "ENZYME",
    "TRANSPORTER",
    "CARRIER",
    "ASSOCIATED_WITH",
]
EMBEDDING_DIM = 128 # Embedding dimensions (128 is standard for biomedical KGs)

class GDSEmbedder:

    def __init__(self, driver: Driver):
        self.driver = driver
    # Public API
    def run(self):
        """
        Full pipeline: project → FastRP → Node2Vec → write-back → drop.
        Call this once with Neo4j running and GDS plugin enabled.
        """
        with self.driver.session() as session:
            self._drop_projection_if_exists(session)
            self._create_projection(session)
            self._run_fastrp(session)
            self._run_node2vec(session)
            self._write_back_fastrp(session)
            self._write_back_node2vec(session)
            self._drop_projection(session)

        print("=== Done. Embeddings stored on all nodes. ===")

    # -------------------------------------------------------------------------
    # Private helpers — Projection
    # -------------------------------------------------------------------------

    def _drop_projection_if_exists(self, session):
        """Drop old projection if it already exists, to avoid errors on re-run."""
        result = session.run(
            "CALL gds.graph.exists($name) YIELD exists",
            name=GRAPH_NAME,
        ).single()

        if result and result["exists"]:
            print(f"  [cleanup] Dropping old projection '{GRAPH_NAME}'...")
            session.run(
                "CALL gds.graph.drop($name) YIELD graphName",
                name=GRAPH_NAME,
            )

    def _create_projection(self, session):
        """
        Load a lightweight in-memory copy of the graph into GDS RAM.
        GDS needs this projection before running any algorithm.
        'UNDIRECTED' tells GDS to treat all edges as bidirectional so
        algorithms like Node2Vec can walk in both directions.
        """

        # Build the relationship projection dict: each rel type → UNDIRECTED
        rel_projection = {rel: {"orientation": "UNDIRECTED"} for rel in REL_TYPES}

        session.run(
            """
            CALL gds.graph.project(
                $name,
                $node_labels,
                $rel_projection
            )
            """,
            name=GRAPH_NAME,
            node_labels=NODE_LABELS,
            rel_projection=rel_projection,
        )
        print("Done")

    def _drop_projection(self, session):
        """Drop the in-memory projection to free RAM after algorithms are done."""

        session.run("CALL gds.graph.drop($name) YIELD graphName", name=GRAPH_NAME)
        print("done")

    # Private helpers — Algorithms

    def _run_fastrp(self, session):
        """
        Run FastRP (Fast Random Projection) on the in-memory projection.
        FastRP works by:
          1. Assigning every node a random initial vector.
          2. Averaging its neighbors' vectors into itself, iteratively.
        Result: a 128-dim embedding stored temporarily inside GDS,
                NOT yet written to the actual Neo4j nodes.
        """

        session.run(
            """
            CALL gds.fastRP.mutate(
                $name,
                {
                    embeddingDimension:   $dim,
                    mutateProperty:       'fastrp_embedding',
                    randomSeed:           42
                }
            )
            """,
            name=GRAPH_NAME,
            dim=EMBEDDING_DIM,
        )
        print(f"Done")

    def _run_node2vec(self, session):
        """
        Run Node2Vec on the in-memory projection.
        Node2Vec works by simulating random walks from every node
        and learning which nodes appear in similar 'neighborhoods'.
        walkLength=80  → each random walk is 80 steps long.
        walksPerNode=10 → we simulate 10 separate walks per node.
        Result: a 128-dim embedding stored temporarily inside GDS.
        """

        session.run(
            """
            CALL gds.node2vec.mutate(
                $name,
                {
                    embeddingDimension:   $dim,
                    walkLength:           80,
                    walksPerNode:         10,
                    mutateProperty:       'node2vec_embedding',
                    randomSeed:           42
                }
            )
            """,
            name=GRAPH_NAME,
            dim=EMBEDDING_DIM,
        )
        print(f"Done")

    # Private helpers — Write-back to Neo4j

    def _write_back_fastrp(self, session):
        """
        The GDS projection holds the embeddings in RAM.
        This step copies them back to the real Neo4j graph
        as a persistent property on each node.
        After this, the embedding survives even if Neo4j restarts.
        """
        
        session.run(
            """
            CALL gds.fastRP.write(
                $name,
                {
                    embeddingDimension: $dim,
                    writeProperty:      'fastrp_embedding',
                    randomSeed:         42
                }
            )
            """,
            name=GRAPH_NAME,
            dim=EMBEDDING_DIM,
        )
        print("fastRP done.")

    def _write_back_node2vec(self, session):
        """
        Same as _write_back_fastrp but for Node2Vec embeddings.
        After this step, every Drug/Gene/Disease node in Neo4j has:
          - node.fastrp_embedding   = [128 floats]
          - node.node2vec_embedding = [128 floats]
        """
        session.run(
            """
            CALL gds.node2vec.write(
                $name,
                {
                    embeddingDimension: $dim,
                    walkLength:         80,
                    walksPerNode:       10,
                    writeProperty:      'node2vec_embedding',
                    randomSeed:         42
                }
            )
            """,
            name=GRAPH_NAME,
            dim=EMBEDDING_DIM,
        )
        print("Node2Vec done.")
