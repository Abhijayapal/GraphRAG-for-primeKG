
from neo4j import Driver

# -- Relationship groups used across all queries -------------------------------
DRUG_GENE_RELS    = "TARGET|ENZYME|TRANSPORTER|CARRIER"
DRUG_DISEASE_RELS = "INDICATION|CONTRAINDICATION|OFF_LABEL_USE"


class CypherRetriever:
    def __init__(self, driver: Driver):
        self.driver = driver

    # Public API — single entity
    def retrieve(
        self,
        node_name: str, #name of entity from database
        #mode->str=it acts like routing switch that tells which type of entity we are asking so that it knowwhich set of cypher query to run
        mode: str,      
        top_k: int = 10, #default value of max results per query
    ) -> list: #return output as list
        
        mode = mode.lower()
        # dispatch table->dictionary of functions->replace if/elif/else blocks
        # it is just for more readability
        dispatch = {
            "disease": self._query_disease,
            "gene":    self._query_gene,
            "drug":    self._query_drug,
        }
        if mode not in dispatch:
            raise ValueError(
                f"mode must be 'disease', 'gene', or 'drug'. "
                f"Got '{mode}'. For combined use retrieve_combined()."
            )
        return dispatch[mode](node_name, top_k)

    # Public API — combined (triplet validation)
    # For multi-hop path validation and reasoning
    # Check all connections between a Drug + Gene + Disease triplet.

    def retrieve_combined(self, drug: str, gene: str, disease: str) -> dict:
        """
        Fires 3 targeted queries, then computes an overall confidence score:
          - all 3 connected  → 1.0  (full_triplet)
          - Drug→Gene + Gene→Disease only  → 0.7  (implicit_via_gene)
          - Drug→Disease only  → 0.9  (direct_only)
          - nothing connected  → 0.0  (no_connection)
        Returns:
            {
                drug, gene, disease,
                drug_disease_rels: list[str],
                drug_gene_rels:    list[str],
                gene_disease_rels: list[str],
                confidence:        float,
                path_type:         str,
                is_contraindication: bool,
            }
        """
        with self.driver.session() as session:

            # Query A: Drug → Disease (any direct relationship)
            # [r] means any relationship
            drug_disease_rows = session.run(
                """
                MATCH (d:Drug {name: $drug})-[r]->(dis:Disease {name: $disease})
                RETURN type(r) AS rel_type
                """,
                drug=drug, disease=disease, #The Cypher query uses them wherever it sees $drug and $disease
            ).data()

            # Query B: Drug → Gene
            drug_gene_rows = session.run(
                f"""
                MATCH (d:Drug {{name: $drug}})-[r:{DRUG_GENE_RELS}]->(g:Gene {{name: $gene}})
                RETURN type(r) AS rel_type
                """,
                drug=drug, gene=gene,
            ).data()

            # Query C: Gene → Disease
            gene_disease_rows = session.run(
                """
                MATCH (g:Gene {name: $gene})-[r:ASSOCIATED_WITH]->(dis:Disease {name: $disease})
                RETURN type(r) AS rel_type
                """,
                gene=gene, disease=disease,
            ).data() # .data() converts the result into list

        # -- Compute confidence --------------------------------------------
        # it is part of cypher ranker but used here to get confidence score for paths
        if drug_disease_rows and drug_gene_rows and gene_disease_rows:
            confidence, path_type = 1.0, "full_triplet"
        elif drug_gene_rows and gene_disease_rows:
            confidence, path_type = 0.7, "implicit_via_gene"
        elif drug_disease_rows:
            confidence, path_type = 0.9, "direct_only"
        else:
            confidence, path_type = 0.0, "no_connection"

        dd_rels = []
        for r in drug_disease_rows:
            dd_rels.append(r["rel_type"])
        return {
            "drug":              drug,
            "gene":              gene,
            "disease":           disease,
            "drug_disease_rels": dd_rels,
            "drug_gene_rels":    [r["rel_type"] for r in drug_gene_rows],
            "gene_disease_rels": [r["rel_type"] for r in gene_disease_rows],
            "confidence":        confidence,
            "path_type":         path_type,
            "is_contraindication": "CONTRAINDICATION" in dd_rels,
        }

    # Private: mode implementations
    def _query_disease(self, disease_name: str, top_k: int) -> list:
        """Disease mode: find candidate Drugs via direct + 1-hop paths."""
        results = []
        with self.driver.session() as session:

            # Query A: direct Drug → Disease 
            # Find Drug that has an INDICATION, CONTRAINDICATION, or OFF_LABEL_USE pointing to disease
            rows = session.run(
                f"""
                MATCH (d:Drug)-[r:{DRUG_DISEASE_RELS}]->(dis:Disease {{name: $name}})
                RETURN d.id AS drug_id, d.name AS drug_name,
                       type(r) AS rel_type, dis.name AS disease_name
                LIMIT $top_k
                """,
                name=disease_name, top_k=top_k,
            ).data()

            for row in rows:
                results.append(self._make_result(
                    candidate_name  = row["drug_name"],
                    candidate_label = "Drug",
                    candidate_id    = row["drug_id"],
                    path_nodes      = [row["drug_name"], disease_name],
                    path_rels       = [row["rel_type"]],
                    path_type       = "direct",
                    rel_type        = row["rel_type"],
                    is_contraindication = (row["rel_type"] == "CONTRAINDICATION"),
                ))

            # Query B: Drug → Gene → Disease (1-hop) 
            # Find me a Drug that targets a Gene, where that Gene is known to cause disease.
            rows = session.run(
                f"""
                MATCH (d:Drug)-[r1:{DRUG_GENE_RELS}]->(g:Gene)
                      -[r2:ASSOCIATED_WITH]->(dis:Disease {{name: $name}})
                RETURN d.id AS drug_id, d.name AS drug_name,
                       type(r1) AS rel1,  g.name AS gene_name,
                       type(r2) AS rel2,  dis.name AS disease_name
                LIMIT $top_k
                """,
                name=disease_name, top_k=top_k,
            ).data()

            for row in rows:
                results.append(self._make_result(
                    candidate_name  = row["drug_name"],
                    candidate_label = "Drug",
                    candidate_id    = row["drug_id"],
                    path_nodes      = [row["drug_name"], row["gene_name"], disease_name],
                    path_rels       = [row["rel1"], row["rel2"]],
                    path_type       = "1-hop",
                    rel_type        = row["rel1"],
                    is_contraindication = False,
                ))

        return results

    # 
    def _query_gene(self, gene_name: str, top_k: int) -> list:
        """Gene mode: find Drugs + Diseases connected to this gene."""
        results = []
        with self.driver.session() as session:

            # Query A: Drugs → Gene  Find any Drug that points to this Gene
            rows = session.run(
                f"""
                MATCH (d:Drug)-[r:{DRUG_GENE_RELS}]->(g:Gene {{name: $name}})
                RETURN d.id AS drug_id, d.name AS drug_name,
                       type(r) AS rel_type, g.name AS gene_name
                LIMIT $top_k
                """,
                name=gene_name, top_k=top_k,
            ).data()

            for row in rows:
                results.append(self._make_result(
                    candidate_name  = row["drug_name"],
                    candidate_label = "Drug",
                    candidate_id    = row["drug_id"],
                    path_nodes      = [row["drug_name"], gene_name],
                    path_rels       = [row["rel_type"]],
                    path_type       = "direct",
                    rel_type        = row["rel_type"],
                    is_contraindication = False,
                ))

            # Query B: Gene → Diseases  Find any Gene that points to this Disease
            rows = session.run(
                """
                MATCH (g:Gene {name: $name})-[r:ASSOCIATED_WITH]->(dis:Disease)
                RETURN dis.id AS dis_id, dis.name AS dis_name,
                       type(r) AS rel_type, g.name AS gene_name
                LIMIT $top_k
                """,
                name=gene_name, top_k=top_k,
            ).data()

            for row in rows:
                results.append(self._make_result(
                    candidate_name  = row["dis_name"],
                    candidate_label = "Disease",
                    candidate_id    = row["dis_id"],
                    path_nodes      = [gene_name, row["dis_name"]],
                    path_rels       = [row["rel_type"]],
                    path_type       = "direct",
                    rel_type        = row["rel_type"],
                    is_contraindication = False,
                ))

        return results

    def _query_drug(self, drug_name: str, top_k: int) -> list:
        """Drug mode: find Diseases + Genes connected to this drug."""
        results = []
        with self.driver.session() as session:

            # Query A: Drug → Diseases  Find any Disease that points to this Drug
            rows = session.run(
                f"""
                MATCH (d:Drug {{name: $name}})-[r:{DRUG_DISEASE_RELS}]->(dis:Disease)
                RETURN dis.id AS dis_id, dis.name AS dis_name,
                       type(r) AS rel_type, d.name AS drug_name
                LIMIT $top_k
                """,
                name=drug_name, top_k=top_k,
            ).data()

            for row in rows:
                results.append(self._make_result(
                    candidate_name  = row["dis_name"],
                    candidate_label = "Disease",
                    candidate_id    = row["dis_id"],
                    path_nodes      = [drug_name, row["dis_name"]],
                    path_rels       = [row["rel_type"]],
                    path_type       = "direct",
                    rel_type        = row["rel_type"],
                    is_contraindication = (row["rel_type"] == "CONTRAINDICATION"),
                ))

            # Query B: Drug → Genes  Find any Gene that points to this Drug
            rows = session.run(
                f"""
                MATCH (d:Drug {{name: $name}})-[r:{DRUG_GENE_RELS}]->(g:Gene)
                RETURN g.id AS gene_id, g.name AS gene_name,
                       type(r) AS rel_type, d.name AS drug_name
                LIMIT $top_k
                """,
                name=drug_name, top_k=top_k,
            ).data()

            for row in rows:
                results.append(self._make_result(
                    candidate_name  = row["gene_name"],
                    candidate_label = "Gene",
                    candidate_id    = row["gene_id"],
                    path_nodes      = [drug_name, row["gene_name"]],
                    path_rels       = [row["rel_type"]],
                    path_type       = "direct",
                    rel_type        = row["rel_type"],
                    is_contraindication = False,
                ))

        return results

    
    # Helper

    @staticmethod
    def _make_result(
        candidate_name, candidate_label, candidate_id,
        path_nodes, path_rels, path_type, rel_type, is_contraindication,
    ) -> dict:
        """Build a standardised result dict."""
        return {
            "candidate_name":    candidate_name,
            "candidate_label":   candidate_label,
            "candidate_id":      candidate_id,
            "path_nodes":        path_nodes,   # list of node names along the path
            "path_rels":         path_rels,    # list of relationship types along the path
            "path_type":         path_type,    # "direct" | "1-hop"
            "rel_type":          rel_type,     # primary relationship type
            "raw_score":         0.0,          # filled in by CypherRanker
            "is_contraindication": is_contraindication,
        }
