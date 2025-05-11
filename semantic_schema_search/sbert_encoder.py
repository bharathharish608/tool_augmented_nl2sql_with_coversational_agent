"""
SBERT embedding and similarity for schema elements.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

class SbertEncoder:
    """
    Handles SBERT encoding and similarity computation for schema elements and queries.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the SBERT encoder with a given model.
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Encode a list of texts (schema elements or queries) into dense vectors.
        Returns a numpy array of shape (len(texts), embedding_dim).
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def similarity(self, query_embedding, schema_embeddings):
        """
        Compute cosine similarity between a query embedding and schema element embeddings.
        Returns a numpy array of similarity scores.
        """
        # Ensure query_embedding is 1D
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]
        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        schema_norms = schema_embeddings / np.linalg.norm(schema_embeddings, axis=1, keepdims=True)
        return np.dot(schema_norms, query_norm) 