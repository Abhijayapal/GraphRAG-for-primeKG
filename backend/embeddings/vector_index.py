"""
VectorIndexBuilder: Creates Neo4j Vector Indexes on top of stored GDS embeddings.

Why we need this:
    After GDSEmbedder writes embeddings onto nodes, we need a way to search them.
    Without an index, Neo4j would have to compare your query vector against
    EVERY single node in the database (brute force — very slow for 100K nodes).
    A Vector Index uses ANN (Approximate Nearest Neighbor) search to find the
    top-K most similar vectors in milliseconds.

Usage:
    builder = VectorIndexBuilder(driver)
    builder.build()   # creates both FastRP + Node2Vec indexes
    results = builder.search_fastrp(query_vector, top_k=10)
"""

from neo4j import Driver


# Index names (used in CREATE INDEX and CALL db.index.vector.queryNodes) 
FASTRP_INDEX_NAME   = "fastrp_index"
NODE2VEC_INDEX_NAME = "node2vec_index"

EMBEDDING_DIM = 128  # Must match the dim used in GDSEmbedder


class VectorIndexBuilder:

    def __init__(self, driver: Driver):
        self.driver = driver

    # Public API

    def build(self):
        """
        Create both Vector Indexes inside Neo4j.
        Safe to run multiple times — drops old indexes first.
        """
        print("=== Vector Index Builder ===")
        with self.driver.session() as session:
            self._drop_index_if_exists(session, FASTRP_INDEX_NAME)
            self._drop_index_if_exists(session, NODE2VEC_INDEX_NAME)
            self._create_fastrp_index(session)
            self._create_node2vec_index(session)
        print("=== Both Vector Indexes created successfully. ===")

    def search_fastrp(self, query_vector: list, top_k: int = 10) -> list:
        """
        Find the top-K most similar nodes to a given query vector,
        using the FastRP Vector Index.

        Args:
            query_vector: A list of 128 floats (the embedding to search with).
            top_k:        Number of similar nodes to return.

        Returns:
            List of dicts: [{"name": ..., "label": ..., "score": ...}, ...]
        """
        with self.driver.session() as session:
            return self._query_index(session, "fastrp_embedding", query_vector, top_k)

    def search_node2vec(self, query_vector: list, top_k: int = 10) -> list:
        """
        Same as search_fastrp but uses the Node2Vec Vector Index.
        """
        with self.driver.session() as session:
            return self._query_index(session, "node2vec_embedding", query_vector, top_k)

    # Private helpers

    def _drop_index_if_exists(self, session, index_name: str):
        """Drop an existing index by name so we can safely recreate it."""
        existing = session.run(
            "SHOW INDEXES WHERE name = $name",
            name=index_name,
        ).data()

        if existing:
            print(f"  [cleanup] Dropping old index '{index_name}'...")
            session.run(f"DROP INDEX {index_name}")

    def _create_fastrp_index(self, session):
        """
        Create a Vector Index on the 'fastrp_embedding' property.
        Neo4j ANN search uses cosine similarity by default.
        After this, Neo4j can respond to similarity queries in milliseconds
        instead of scanning all 100K+ nodes one by one.
        """
        print(f"  [index] Creating '{FASTRP_INDEX_NAME}' on fastrp_embedding...")
        session.run(
            f"""
            CREATE VECTOR INDEX {FASTRP_INDEX_NAME}
            FOR (n:Drug|Gene|Disease)
            ON n.fastrp_embedding
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`:        {EMBEDDING_DIM},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
        )
        print(f"  [index] '{FASTRP_INDEX_NAME}' created.")

    def _create_node2vec_index(self, session):
        """
        Same as _create_fastrp_index but targets node2vec_embedding.
        """
        print(f"  [index] Creating '{NODE2VEC_INDEX_NAME}' on node2vec_embedding...")
        session.run(
            f"""
            CREATE VECTOR INDEX {NODE2VEC_INDEX_NAME}
            FOR (n:Drug|Gene|Disease)
            ON n.node2vec_embedding
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`:          {EMBEDDING_DIM},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
        )
        print(f"  [index] '{NODE2VEC_INDEX_NAME}' created.")

    def _query_index(
        self,
        session,
        embedding_property: str,
        query_vector: list,
        top_k: int,
    ) -> list:
        """
        Run a KNN (K-Nearest Neighbor) vector similarity search.
        Neo4j computes cosine similarity between query_vector and all
        stored embeddings using the ANN index, then returns the top-K matches.
        """
        rows = session.run(
            f"""
            MATCH (n)
            WHERE n.{embedding_property} IS NOT NULL
            WITH n, vector.similarity.cosine(n.{embedding_property}, $query_vector) AS score
            ORDER BY score DESC
            LIMIT $top_k
            RETURN
                n.name  AS name,
                labels(n)[0] AS label,
                score
            """,
            top_k=top_k,
            query_vector=query_vector,
        ).data()

        return rows
