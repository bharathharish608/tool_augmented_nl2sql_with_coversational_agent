"""
BM25 indexing and retrieval for schema elements.
"""

from rank_bm25 import BM25Okapi
from semantic_schema_search.utils import load_schema
from pathlib import Path

class BM25Index:
    """
    Handles BM25 indexing and retrieval for a schema corpus.
    """
    def __init__(self, schema_corpus):
        """
        Initialize the BM25 index with a list of schema element texts.
        """
        self.schema_corpus = schema_corpus
        self.tokenized_corpus = [doc.lower().split() for doc in schema_corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=10):
        """
        Search the schema corpus for the query and return top_k results ranked by BM25 score.
        Returns a list of (schema_element, score) tuples.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self.schema_corpus, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

if __name__ == "__main__":
    # Example usage: build index and search
    schema_path = Path(__file__).parent.parent / "global" / "tpcds_with_all_descriptions.json"
    schema_corpus = load_schema(schema_path)
    bm25_index = BM25Index(schema_corpus)
    sample_query = "users"
    print(f"Top 5 BM25 results for query: '{sample_query}'\n")
    results = bm25_index.search(sample_query, top_k=5)
    for i, (element, score) in enumerate(results, 1):
        print(f"{i}. [{score:.2f}] {element}") 