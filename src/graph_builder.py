"""Graph construction: payment data → directed weighted graphs per quarter.

Implements:
- Eq. 1: Adjacency matrix A_t
- Eq. 2: Row-normalized adjacency matrix
- Eq. 8: Multi-hop (2-hop) connection matrix B_t = A_t^2
"""

import networkx as nx
import numpy as np
import pandas as pd


def build_quarterly_graphs(df):
    """Build one directed weighted graph per quarter from payment data.

    Args:
        df: DataFrame with columns [source, target, value, quarter].

    Returns:
        Dict mapping quarter string → nx.DiGraph with edge attribute 'weight'.
    """
    graphs = {}
    for quarter, group in df.groupby("quarter"):
        G = nx.DiGraph()
        # Add all unique industries as nodes
        all_industries = set(group["source"].unique()) | set(group["target"].unique())
        G.add_nodes_from(all_industries)

        # Aggregate payments per (source, target) pair within the quarter
        agg = group.groupby(["source", "target"])["value"].sum().reset_index()
        for _, row in agg.iterrows():
            G.add_edge(row["source"], row["target"], weight=row["value"])

        graphs[quarter] = G

    # Ensure consistent node ordering across quarters
    all_nodes = set()
    for G in graphs.values():
        all_nodes.update(G.nodes())
    all_nodes = sorted(all_nodes)

    for quarter, G in graphs.items():
        for node in all_nodes:
            if node not in G:
                G.add_node(node)

    return graphs


def get_node_list(graphs):
    """Get a consistent sorted list of all nodes across all quarters."""
    all_nodes = set()
    for G in graphs.values():
        all_nodes.update(G.nodes())
    return sorted(all_nodes)


def build_adjacency_matrix(G, node_list=None):
    """Convert graph to dense adjacency matrix A_t (Eq. 1).

    Args:
        G: nx.DiGraph with 'weight' edge attribute.
        node_list: Ordered list of nodes. If None, uses sorted G.nodes().

    Returns:
        numpy array of shape (n, n) where A[i,j] = w_{ij}.
    """
    if node_list is None:
        node_list = sorted(G.nodes())
    n = len(node_list)
    node_idx = {node: i for i, node in enumerate(node_list)}

    A = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        A[i, j] = data.get("weight", 1.0)

    return A


def row_normalize(A):
    """Row-normalize adjacency matrix (Eq. 2).

    A_bar[i,j] = A[i,j] / sum_k(A[i,k])

    Converts absolute payment values into proportional allocations.
    Rows with zero sum (isolated nodes) remain all zeros.

    Returns:
        Row-normalized matrix of same shape.
    """
    row_sums = A.sum(axis=1, keepdims=True)
    # Avoid division by zero for isolated nodes
    row_sums[row_sums == 0] = 1.0
    return A / row_sums


def compute_multihop_matrix(A):
    """Compute 2-hop connection matrix B_t = A_t^2 (Eq. 8).

    B[i,j] = sum_k A[i,k] * A[k,j]
    Represents weighted sum of all two-step payment paths from i to j.

    Args:
        A: Adjacency matrix (can be raw or normalized).

    Returns:
        Matrix of same shape representing indirect connectivity.
    """
    return A @ A


def get_graph_summary(G):
    """Get basic summary statistics for a graph."""
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "total_weight": sum(d["weight"] for _, _, d in G.edges(data=True)),
    }
