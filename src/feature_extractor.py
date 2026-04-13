"""Graph-theoretic feature extraction.

Implements:
- Eq. 3: Betweenness centrality
- Eq. 4: In-degree and out-degree centrality
- Eq. 5: Local clustering coefficient
- Eq. 6: Network density
- Eq. 7: Average path length
"""

import networkx as nx
import numpy as np
import pandas as pd


def extract_node_features(G):
    """Extract node-level graph-theoretic features for all nodes.

    Features per node:
    - in_degree_centrality, out_degree_centrality (Eq. 4)
    - in_strength, out_strength (weighted degree)
    - betweenness_centrality (Eq. 3)
    - eigenvector_centrality
    - clustering_coefficient (Eq. 5)

    Args:
        G: nx.DiGraph with 'weight' edge attribute.

    Returns:
        DataFrame indexed by node name with feature columns.
    """
    nodes = sorted(G.nodes())
    features = {}

    # Degree centrality (Eq. 4)
    in_deg = nx.in_degree_centrality(G)
    out_deg = nx.out_degree_centrality(G)

    # Weighted degree (strength)
    in_strength = dict(G.in_degree(weight="weight"))
    out_strength = dict(G.out_degree(weight="weight"))

    # Betweenness centrality (Eq. 3)
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)

    # Eigenvector centrality (using numpy solver for robustness)
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eigenvector = {n: 0.0 for n in nodes}

    # Clustering coefficient (Eq. 5)
    clustering = nx.clustering(G)

    for node in nodes:
        features[node] = {
            "in_degree_centrality": in_deg.get(node, 0),
            "out_degree_centrality": out_deg.get(node, 0),
            "in_strength": in_strength.get(node, 0),
            "out_strength": out_strength.get(node, 0),
            "betweenness_centrality": betweenness.get(node, 0),
            "eigenvector_centrality": eigenvector.get(node, 0),
            "clustering_coefficient": clustering.get(node, 0),
        }

    return pd.DataFrame.from_dict(features, orient="index")


def extract_network_features(G):
    """Extract global network topology features for a graph.

    Features:
    - density (Eq. 6)
    - average_path_length (Eq. 7) — computed on largest SCC
    - num_edges
    - average_clustering

    Args:
        G: nx.DiGraph.

    Returns:
        Dict of global network metrics.
    """
    n = G.number_of_nodes()
    features = {
        "density": nx.density(G),
        "num_edges": G.number_of_edges(),
        "num_nodes": n,
        "average_clustering": nx.average_clustering(G),
    }

    # Average path length on largest strongly connected component
    # (full graph may be disconnected)
    if n > 1:
        sccs = list(nx.strongly_connected_components(G))
        largest_scc = max(sccs, key=len)
        if len(largest_scc) > 1:
            subgraph = G.subgraph(largest_scc)
            features["average_path_length"] = nx.average_shortest_path_length(subgraph)
            features["scc_size"] = len(largest_scc)
        else:
            features["average_path_length"] = float("inf")
            features["scc_size"] = 1
    else:
        features["average_path_length"] = 0.0
        features["scc_size"] = n

    return features


def extract_edge_features(G, node_features_df, multihop_matrix, node_list):
    """Build edge-level feature matrix combining source/target node features and 2-hop strength.

    For each edge (i, j), the feature vector includes:
    - Source node's graph features (prefixed 'src_')
    - Target node's graph features (prefixed 'tgt_')
    - 2-hop connection strength B[i,j]
    - Edge weight

    Args:
        G: nx.DiGraph.
        node_features_df: DataFrame from extract_node_features.
        multihop_matrix: B = A^2 matrix from graph_builder.compute_multihop_matrix.
        node_list: Ordered list of node names matching matrix indices.

    Returns:
        DataFrame with one row per edge and feature columns.
    """
    node_idx = {node: i for i, node in enumerate(node_list)}
    rows = []

    for u, v, data in G.edges(data=True):
        row = {"source": u, "target": v, "weight": data.get("weight", 0)}

        # Source node features
        if u in node_features_df.index:
            for col in node_features_df.columns:
                row[f"src_{col}"] = node_features_df.loc[u, col]

        # Target node features
        if v in node_features_df.index:
            for col in node_features_df.columns:
                row[f"tgt_{col}"] = node_features_df.loc[v, col]

        # 2-hop connection strength
        i, j = node_idx.get(u), node_idx.get(v)
        if i is not None and j is not None:
            row["multihop_strength"] = multihop_matrix[i, j]
        else:
            row["multihop_strength"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def extract_all_features(graphs, node_list, adjacency_matrices):
    """Extract all features for all quarters.

    Args:
        graphs: Dict[quarter_str, nx.DiGraph].
        node_list: Consistent ordered list of node names.
        adjacency_matrices: Dict[quarter_str, np.ndarray] (raw adjacency).

    Returns:
        Tuple of:
        - node_features: Dict[quarter_str, DataFrame]
        - network_features: Dict[quarter_str, dict]
        - edge_features: Dict[quarter_str, DataFrame]
    """
    from src.graph_builder import compute_multihop_matrix, row_normalize

    all_node_features = {}
    all_network_features = {}
    all_edge_features = {}

    for quarter in sorted(graphs.keys()):
        G = graphs[quarter]
        A = adjacency_matrices[quarter]

        # Node-level features
        nf = extract_node_features(G)
        all_node_features[quarter] = nf

        # Network-level features
        net_f = extract_network_features(G)
        all_network_features[quarter] = net_f

        # Edge-level features with 2-hop
        A_norm = row_normalize(A)
        B = compute_multihop_matrix(A_norm)
        ef = extract_edge_features(G, nf, B, node_list)
        ef["quarter"] = quarter

        # Add network-level features to each edge
        for key, val in net_f.items():
            if isinstance(val, (int, float)):
                ef[f"net_{key}"] = val

        all_edge_features[quarter] = ef

    return all_node_features, all_network_features, all_edge_features
