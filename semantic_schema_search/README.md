# semantic_schema_search

This module provides semantic schema search and retrieval for NL2SQL and related pipelines, using a fusion of BM25 and SBERT for robust, synonym-aware schema element retrieval.

## Planned Structure

- `bm25_index.py`: BM25 indexing and retrieval logic
- `sbert_encoder.py`: SBERT embedding and similarity logic
- `fusion_search.py`: Score fusion and search API
- `train_fusion.py`: Training/tuning script for SBERT and fusion weights
- `evaluate.py`: Evaluation metrics and scripts
- `utils.py`: Shared helpers (data loading, normalization, etc.)
- `requirements.txt`: Dependencies

## Usage

Import the fusion search API in your NL2SQL pipeline:

```python
from semantic_schema_search.fusion_search import semantic_search
```

Call `semantic_search(query)` to retrieve relevant schema elements for a natural language query. 