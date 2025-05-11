import json
from pathlib import Path
import numpy as np

SCHEMA_PATH = Path(__file__).parent.parent / "global" / "tpcds_with_all_descriptions.json"

class SchemaTools:
    def __init__(self, schema_path=SCHEMA_PATH):
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        self.tables = self.schema["tables"]

    def list_tables(self):
        """Return a list of all table names in the schema."""
        return list(self.tables.keys())

    def get_columns(self, table_name):
        """Return a list of columns (and descriptions if available) for a given table."""
        table = self.tables.get(table_name)
        if not table:
            raise ValueError(f"Table '{table_name}' not found in schema.")
        columns = table.get('columns', {})
        if isinstance(columns, dict):
            return [(col, desc.get('description') if isinstance(desc, dict) else desc) for col, desc in columns.items()]
        elif isinstance(columns, list):
            return [(col, None) for col in columns]
        else:
            raise ValueError(f"Unexpected columns format for table '{table_name}'.")

    def search_schema(self, keyword):
        """Return tables and columns matching the keyword (case-insensitive, matches in table/column names or descriptions)."""
        keyword = keyword.lower()
        results = []
        for table_name, table in self.tables.items():
            # Check table name
            if keyword in table_name.lower():
                results.append({'table': table_name, 'column': None, 'description': table.get('description')})
            columns = table.get('columns', {})
            if isinstance(columns, dict):
                for col, desc in columns.items():
                    col_desc = desc.get('description') if isinstance(desc, dict) else desc
                    if (keyword in col.lower()) or (col_desc and keyword in str(col_desc).lower()):
                        results.append({'table': table_name, 'column': col, 'description': col_desc})
            elif isinstance(columns, list):
                for col in columns:
                    if keyword in col.lower():
                        results.append({'table': table_name, 'column': col, 'description': None})
        return results

    def get_column_description(self, table_name, column_name):
        """Return the description for a specific column, or None if not available."""
        table = self.tables.get(table_name)
        if not table:
            raise ValueError(f"Table '{table_name}' not found in schema.")
        columns = table.get('columns', {})
        if isinstance(columns, dict):
            desc = columns.get(column_name)
            if isinstance(desc, dict):
                return desc.get('description')
            return desc
        elif isinstance(columns, list):
            return None if column_name not in columns else None
        else:
            raise ValueError(f"Unexpected columns format for table '{table_name}'.")

    def semantic_search_schema(self, keyword, top_k=10, alpha=0.5):
        """
        Return top-k semantically matching schema elements for a keyword using BM25+SBERT fusion.
        """
        from semantic_schema_search.bm25_index import BM25Index
        from semantic_schema_search.sbert_encoder import SbertEncoder
        from semantic_schema_search.utils import load_schema, normalize_scores
        schema_path = Path(__file__).parent.parent / "global" / "tpcds_with_all_descriptions.json"
        if not hasattr(self, '_semantic_cache'):
            schema_corpus = load_schema(schema_path)
            self._semantic_cache = {
                'schema_corpus': schema_corpus,
                'bm25_index': BM25Index(schema_corpus),
                'sbert_encoder': SbertEncoder()
            }
        schema_corpus = self._semantic_cache['schema_corpus']
        bm25_index = self._semantic_cache['bm25_index']
        sbert_encoder = self._semantic_cache['sbert_encoder']
        bm25_scores = dict(bm25_index.search(keyword, top_k=len(schema_corpus)))
        schema_embeddings = sbert_encoder.encode(schema_corpus)
        query_emb = sbert_encoder.encode([keyword])
        sbert_scores_arr = sbert_encoder.similarity(query_emb, schema_embeddings)
        sbert_scores = {elem: sbert_scores_arr[i] for i, elem in enumerate(schema_corpus)}
        bm25_vals = np.array(list(bm25_scores.values()))
        sbert_vals = np.array([sbert_scores[elem] for elem in bm25_scores.keys()])
        bm25_norm = normalize_scores(bm25_vals)
        sbert_norm = normalize_scores(sbert_vals)
        fused_scores = alpha * bm25_norm + (1 - alpha) * sbert_norm
        fused_dict = {elem: fused_scores[i] for i, elem in enumerate(bm25_scores.keys())}
        ranked = sorted(fused_dict.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def relation_aware_retrieval(self, nl_query, top_k=10):
        """
        Return top-k relevant schema elements using relation-aware retrieval (context-aware RWR).
        """
        from relation_aware_retrieval.relation_aware_tool import RelationAwareRetriever
        schema_json_path = str(Path(__file__).parent.parent / "global" / "tpcds_with_all_descriptions.json")
        labelled_csv_path = str(Path(__file__).parent.parent / "labelled_dataset_gen" / "datasets" / "labelled_dataset.csv")
        retriever = RelationAwareRetriever(schema_json_path, labelled_csv_path)
        return retriever.retrieve(nl_query, top_k=top_k)

# Example usage
if __name__ == "__main__":
    tools = SchemaTools()
    print("Tables in schema:")
    print(tools.list_tables())
    print("\nColumns in first table:")
    first_table = tools.list_tables()[0]
    print(f"Table: {first_table}")
    print(tools.get_columns(first_table))
    print("\nSearch for 'sales':")
    print(tools.search_schema('sales'))
    print("\nDescription for first column in first table:")
    first_col = tools.get_columns(first_table)[0][0]
    print(f"{first_table}.{first_col}: {tools.get_column_description(first_table, first_col)}") 