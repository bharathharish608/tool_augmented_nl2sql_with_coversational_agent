from semantic_schema_search.utils import load_schema
from semantic_schema_search.bm25_index import BM25Index
from pathlib import Path

if __name__ == "__main__":
    schema_path = Path(__file__).parent.parent / "global" / "tpcds_with_all_descriptions.json"
    schema_corpus = load_schema(schema_path)
    print(f"Loaded {len(schema_corpus)} schema elements.")
    bm25_index = BM25Index(schema_corpus)
    sample_query = "users"
    print(f"\nTop 10 BM25 results for query: '{sample_query}'\n")
    results = bm25_index.search(sample_query, top_k=10)
    for i, (element, score) in enumerate(results, 1):
        print(f"{i}. [{score:.2f}] {element}") 