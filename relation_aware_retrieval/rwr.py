import numpy as np
import networkx as nx

def context_aware_rwr(G, context_nodes, alpha=0.15, max_iter=100, tol=1e-6, weight_attr='weight'):
    """
    Run context-aware RWR on graph G, starting from context_nodes.
    Returns a dict of node: stationary probability.
    """
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    # Personalized restart vector: 1 for context nodes, 0 otherwise, normalized
    p0 = np.zeros(n)
    for node in context_nodes:
        if node in idx:
            p0[idx[node]] = 1
    if p0.sum() == 0:
        p0[:] = 1  # fallback to uniform
    p0 /= p0.sum()
    # Build weighted adjacency matrix
    A = np.zeros((n, n))
    for i, u in enumerate(nodes):
        neighbors = list(G.neighbors(u))
        weights = np.array([G[u][v].get(weight_attr, 1.0) for v in neighbors])
        if weights.sum() == 0:
            continue
        weights = weights / weights.sum()
        for j, v in enumerate(neighbors):
            A[idx[v], i] = weights[j]
    p = p0.copy()
    for _ in range(max_iter):
        p_new = alpha * p0 + (1 - alpha) * A @ p
        if np.linalg.norm(p_new - p, 1) < tol:
            break
        p = p_new
    return {nodes[i]: p[i] for i in range(n)} 