class PathExtractor:    
    # A result dict from CypherRetriever with 'path_nodes' and 'path_rels'.
    def format(self, result: dict) -> str:
        nodes = result.get("path_nodes", [])
        rels  = result.get("path_rels",  [])

        if not nodes: raise ValueError("path_nodes cannot be empty.")

        # Single node — no relationships
        if len(nodes) == 1: return nodes[0]

        # Interleave nodes and rels: node → rel → node → rel → node ...
        parts = [nodes[0]]
        for rel, node in zip(rels, nodes[1:]):
            parts.append(rel)
            parts.append(node)

        return " → ".join(parts)

    # Function use -> Format a list of result dicts into a list of path strings.
    # Input -> results: List of result dicts from CypherRetriever.
    # Output -> List of (result_dict, path_string) tuples.
    def format_all(self, results: list) -> list:
        output = []
        for res in results:
            try:
                path_str = self.format(res)
            except ValueError:
                path_str = "(no path)"
            output.append((res, path_str))
        return output

    # function use:-Format a combined-mode result (from retrieve_combined) into a multi-line summary string.
    # Input -> combined_result: Dict returned by CypherRetriever.retrieve_combined().
    # Output -> Multi-line string summarising all connections.
    def format_combined(self, combined_result: dict) -> str:
        drug    = combined_result["drug"]
        gene    = combined_result["gene"]
        disease = combined_result["disease"]
        conf    = combined_result["confidence"]
        ptype   = combined_result["path_type"]

        lines = [
            f"Query: {drug} + {gene} + {disease}",
            f"Confidence: {conf}  |  Path type: {ptype}",
            "",
        ]

        # Drug → Disease
        dd_rels = combined_result["drug_disease_rels"]
        if dd_rels:
            for rel in dd_rels:
                flag = "  *** CONTRAINDICATION ***" if rel == "CONTRAINDICATION" else ""
                lines.append(f"  {drug} → {rel} → {disease}{flag}")
        else:
            lines.append(f"  {drug} -/→ {disease}  (no direct connection)")

        # Drug → Gene
        dg_rels = combined_result["drug_gene_rels"]
        if dg_rels:
            for rel in dg_rels:
                lines.append(f"  {drug} → {rel} → {gene}")
        else:
            lines.append(f"  {drug} -/→ {gene}  (no connection)")

        # Gene → Disease
        gd_rels = combined_result["gene_disease_rels"]
        if gd_rels:
            for rel in gd_rels:
                lines.append(f"  {gene} → {rel} → {disease}")
        else:
            lines.append(f"  {gene} -/→ {disease}  (no connection)")

        return "\n".join(lines)
