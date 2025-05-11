"""
Training and tuning script for SBERT fine-tuning and BM25+SBERT fusion weight optimization.
"""

from semantic_schema_search.utils import load_schema, load_labelled_pairs, normalize_scores
from semantic_schema_search.bm25_index import BM25Index
from semantic_schema_search.sbert_encoder import SbertEncoder
import numpy as np
from pathlib import Path
from collections import defaultdict

def train_sbert(labelled_pairs, schema_corpus, model_name='all-MiniLM-L6-v2'):
    """
    Fine-tune SBERT on labelled (query, relevant/irrelevant schema element) pairs.
    """
    pass

def tune_fusion_weights(labelled_pairs, bm25_index, sbert_encoder, schema_corpus, alphas=None, top_k=10):
    """
    Tune the fusion parameter alpha using labelled data and retrieval metrics.
    """
    if alphas is None:
        alphas = np.linspace(0, 1, 11)
    # Build lookup for schema element index
    schema_idx = {elem: i for i, elem in enumerate(schema_corpus)}
    # Group pairs by query
    query_to_pairs = defaultdict(list)
    for query, schema_elem, label in labelled_pairs:
        query_to_pairs[query].append((schema_elem, label))
    # Precompute SBERT embeddings
    schema_embeddings = sbert_encoder.encode(schema_corpus)
    results = {}
    for alpha in alphas:
        ranks = []
        for query, pairs in query_to_pairs.items():
            # Get gold positives
            gold = set([s for s, l in pairs if l == 1])
            if not gold:
                continue
            # BM25 scores
            bm25_scores = dict(bm25_index.search(query, top_k=len(schema_corpus)))
            # SBERT scores
            query_emb = sbert_encoder.encode([query])
            sbert_scores_arr = sbert_encoder.similarity(query_emb, schema_embeddings)
            sbert_scores = {elem: sbert_scores_arr[i] for elem, i in schema_idx.items()}
            # Normalize scores
            bm25_vals = np.array(list(bm25_scores.values()))
            sbert_vals = np.array(list(sbert_scores.values()))
            bm25_norm = normalize_scores(bm25_vals)
            sbert_norm = normalize_scores(sbert_vals)
            fused_scores = alpha * bm25_norm + (1 - alpha) * sbert_norm
            # Map back to schema elements
            fused_dict = {elem: fused_scores[i] for i, elem in enumerate(bm25_scores.keys())}
            # Rank all schema elements by fused score
            ranked = sorted(fused_dict.items(), key=lambda x: x[1], reverse=True)
            ranked_elems = [elem for elem, _ in ranked]
            # Compute reciprocal rank for any gold
            rr = 0
            for rank, elem in enumerate(ranked_elems[:top_k], 1):
                if elem in gold:
                    rr = 1.0 / rank
                    break
            ranks.append(rr)
        mrr = np.mean(ranks) if ranks else 0.0
        results[alpha] = mrr
        print(f"Alpha={alpha:.2f} | MRR@{top_k}: {mrr:.4f}")
    best_alpha = max(results, key=results.get)
    print(f"\nBest fusion alpha: {best_alpha:.2f} (MRR@{top_k}: {results[best_alpha]:.4f})")
    return best_alpha, results

if __name__ == "__main__":
    # Paths
    schema_path = Path(__file__).parent.parent / "global" / "tpcds_with_all_descriptions.json"
    labelled_path = Path(__file__).parent.parent / "labelled_dataset_gen" / "datasets" / "labelled_dataset.csv"
    # Load data
    schema_corpus = load_schema(schema_path)
    labelled_pairs = load_labelled_pairs(labelled_path, schema_corpus=schema_corpus)
    print(f"Loaded {len(schema_corpus)} schema elements and {len(labelled_pairs)} labelled pairs.")

    # Debug: print a few labelled pairs
    print("\nSample labelled pairs:")
    for i, (query, schema_elem, label) in enumerate(labelled_pairs[:5]):
        print(f"[{i}] Query: {query}\n    Schema: {schema_elem}\n    Label: {label}")
    # Debug: print a few schema elements
    print("\nSample schema_corpus elements:")
    for i, elem in enumerate(schema_corpus[:5]):
        print(f"[{i}] {elem}")

    # Build BM25 and SBERT
    bm25_index = BM25Index(schema_corpus)
    sbert_encoder = SbertEncoder()

    # Debug: For the first query, print gold and top-10 retrieved
    query_to_pairs = defaultdict(list)
    for query, schema_elem, label in labelled_pairs:
        query_to_pairs[query].append((schema_elem, label))
    first_query = next(iter(query_to_pairs))
    gold = set([s for s, l in query_to_pairs[first_query] if l == 1])
    print(f"\nFirst query: {first_query}")
    print(f"Gold schema elements: {gold}")
    bm25_scores = dict(bm25_index.search(first_query, top_k=len(schema_corpus)))
    schema_embeddings = sbert_encoder.encode(schema_corpus)
    query_emb = sbert_encoder.encode([first_query])
    sbert_scores_arr = sbert_encoder.similarity(query_emb, schema_embeddings)
    sbert_scores = {elem: sbert_scores_arr[i] for i, elem in enumerate(schema_corpus)}
    bm25_vals = np.array(list(bm25_scores.values()))
    sbert_vals = np.array(list(sbert_scores.values()))
    bm25_norm = normalize_scores(bm25_vals)
    sbert_norm = normalize_scores(sbert_vals)
    alpha = 0.5
    fused_scores = alpha * bm25_norm + (1 - alpha) * sbert_norm
    fused_dict = {elem: fused_scores[i] for i, elem in enumerate(bm25_scores.keys())}
    ranked = sorted(fused_dict.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 fused retrievals for first query:")
    for i, (elem, score) in enumerate(ranked[:10], 1):
        print(f"{i}. [{score:.2f}] {elem} {'<-- GOLD' if elem in gold else ''}")

    # Tune fusion
    tune_fusion_weights(labelled_pairs, bm25_index, sbert_encoder, schema_corpus) 