"""Interactive network graph rendering using Plotly."""

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc

# Industries to exclude from visualisation
EXCLUDED_INDUSTRIES = {"Unknown or Unclassified"}

# Available edge-colouring metrics.
# Each entry maps a UI label to (description, source_type).
# source_type is "source_node", "target_node", or "edge" to indicate
# where the metric comes from.
EDGE_COLOR_METRICS = {
    "Payment Volume (weight)": {
        "key": "weight",
        "source": "edge",
        "description": "Raw payment value on this edge",
    },
    "Source Out-Strength": {
        "key": "out_strength",
        "source": "source_node",
        "description": "Total outgoing payment volume of the source industry",
    },
    "Target In-Strength": {
        "key": "in_strength",
        "source": "target_node",
        "description": "Total incoming payment volume of the target industry",
    },
    "Source Betweenness Centrality": {
        "key": "betweenness_centrality",
        "source": "source_node",
        "description": "How often the source industry lies on shortest payment paths",
    },
    "Target Betweenness Centrality": {
        "key": "betweenness_centrality",
        "source": "target_node",
        "description": "How often the target industry lies on shortest payment paths",
    },
    "Source Out-Degree Centrality": {
        "key": "out_degree_centrality",
        "source": "source_node",
        "description": "Fraction of industries the source pays",
    },
    "Target In-Degree Centrality": {
        "key": "in_degree_centrality",
        "source": "target_node",
        "description": "Fraction of industries that pay the target",
    },
    "Source Eigenvector Centrality": {
        "key": "eigenvector_centrality",
        "source": "source_node",
        "description": "Connection quality — linked to other important industries",
    },
    "Source Clustering Coefficient": {
        "key": "clustering_coefficient",
        "source": "source_node",
        "description": "How tightly the source's trading partners trade with each other",
    },
}

# Colour scales suitable for continuous edge metrics
EDGE_COLORSCALES = {
    "Viridis": "Viridis",
    "Plasma": "Plasma",
    "Inferno": "Inferno",
    "Turbo": "Turbo",
    "RdYlBu": "RdYlBu",
    "YlOrRd": "YlOrRd",
    "Blues": "Blues",
}


def filter_graph(G):
    """Return a copy of G with excluded industries removed."""
    nodes_to_remove = [n for n in G.nodes() if n in EXCLUDED_INDUSTRIES]
    if not nodes_to_remove:
        return G
    H = G.copy()
    H.remove_nodes_from(nodes_to_remove)
    return H


def compute_stable_layout(graphs, node_list, seed=42, k=0.3):
    """Compute stable node positions driven by payment-flow closeness.

    Nodes with strong mutual payment flows are pulled closer together;
    weakly connected nodes drift apart.  Uses spring_layout on the
    symmetrised, log-scaled aggregated graph so edge weight acts as an
    attraction force — heavier flows => shorter distance.

    Args:
        graphs: Dict[quarter_str, nx.DiGraph].
        node_list: Ordered list of node names.
        seed: Random seed for reproducibility.
        k: Base optimal distance between nodes (spring_layout parameter).

    Returns:
        Dict mapping node_name -> (x, y) position.
    """
    # Filter excluded industries from node list
    node_list = [n for n in node_list if n not in EXCLUDED_INDUSTRIES]

    # Build aggregated directed graph
    agg = nx.DiGraph()
    agg.add_nodes_from(node_list)
    for G in graphs.values():
        for u, v, data in G.edges(data=True):
            if u in EXCLUDED_INDUSTRIES or v in EXCLUDED_INDUSTRIES:
                continue
            if agg.has_edge(u, v):
                agg[u][v]["weight"] += data.get("weight", 1)
            else:
                agg.add_edge(u, v, weight=data.get("weight", 1))

    # Symmetrise: use max(w_ij, w_ji) so mutual flows pull nodes together
    sym = nx.Graph()
    sym.add_nodes_from(node_list)
    for u, v, data in agg.edges(data=True):
        w_uv = data["weight"]
        w_vu = agg[v][u]["weight"] if agg.has_edge(v, u) else 0
        w = max(w_uv, w_vu)
        if sym.has_edge(u, v):
            sym[u][v]["weight"] = max(sym[u][v]["weight"], w)
        else:
            sym.add_edge(u, v, weight=w)

    # Log-scale weights so a few giant flows don't dominate the layout
    for u, v, data in sym.edges(data=True):
        data["weight"] = np.log1p(data["weight"])

    pos = nx.spring_layout(
        sym,
        k=k,
        seed=seed,
        iterations=200,
        weight="weight",   # heavier edges -> stronger attraction
    )
    return pos


def _get_edge_metric_values(G, edges, node_features_df, metric_name):
    """Compute the colouring metric for each edge.

    Args:
        G: nx.DiGraph (already filtered).
        edges: List of (u, v, weight) tuples.
        node_features_df: DataFrame of node features indexed by node name.
        metric_name: Key into EDGE_COLOR_METRICS.

    Returns:
        List of float values, one per edge, in the same order as *edges*.
    """
    info = EDGE_COLOR_METRICS.get(metric_name)
    if info is None:
        # Fallback: uniform grey
        return [1.0] * len(edges)

    key = info["key"]
    source = info["source"]

    values = []
    for u, v, w in edges:
        if source == "edge":
            values.append(w)
        elif source == "source_node":
            val = 0.0
            if node_features_df is not None and u in node_features_df.index:
                val = node_features_df.loc[u, key] if key in node_features_df.columns else 0.0
            values.append(float(val))
        elif source == "target_node":
            val = 0.0
            if node_features_df is not None and v in node_features_df.index:
                val = node_features_df.loc[v, key] if key in node_features_df.columns else 0.0
            values.append(float(val))
        else:
            values.append(0.0)
    return values


def _sample_colorscale(name, t):
    """Sample a Plotly colorscale at normalised position t in [0, 1]."""
    return pc.sample_colorscale(name, [t])[0]


def create_network_figure(
    G,
    pos,
    node_categories,
    category_colors,
    title="",
    max_edges=300,
    node_size_range=(8, 40),
    edge_width_range=(0.3, 3.0),
    edge_color_metric="Payment Volume (weight)",
    edge_colorscale="Viridis",
    node_features_df=None,
):
    """Create a Plotly figure of the payment network graph.

    Args:
        G: nx.DiGraph for the selected quarter.
        pos: Dict of node -> (x, y) positions.
        node_categories: Dict of node_name -> category string.
        category_colors: Dict of category -> hex color.
        title: Figure title.
        max_edges: Maximum number of edges to display (top by weight).
        node_size_range: (min_size, max_size) for node markers.
        edge_width_range: (min_width, max_width) for edge lines.
        edge_color_metric: Name from EDGE_COLOR_METRICS for colouring edges.
        edge_colorscale: Plotly colourscale name.
        node_features_df: DataFrame of node features (needed for metric edges).

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # --- Filter out excluded industries ---
    G = filter_graph(G)

    # --- Edge traces ---
    edges = [(u, v, d.get("weight", 0)) for u, v, d in G.edges(data=True)]
    edges.sort(key=lambda x: x[2], reverse=True)
    edges = edges[:max_edges]

    if edges:
        max_w = max(e[2] for e in edges)
        min_w = min(e[2] for e in edges)
        w_range = max_w - min_w if max_w > min_w else 1

        # Compute colour metric values
        color_values = _get_edge_metric_values(G, edges, node_features_df, edge_color_metric)
        c_min = min(color_values) if color_values else 0
        c_max = max(color_values) if color_values else 1
        c_range = c_max - c_min if c_max > c_min else 1

        # Draw each edge
        for idx, (u, v, w) in enumerate(edges):
            x0, y0 = pos.get(u, (0, 0))
            x1, y1 = pos.get(v, (0, 0))
            # Normalize width by payment weight
            norm_w = (w - min_w) / w_range
            width = edge_width_range[0] + norm_w * (edge_width_range[1] - edge_width_range[0])
            # Normalize colour
            norm_c = (color_values[idx] - c_min) / c_range
            color = _sample_colorscale(edge_colorscale, norm_c)

            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="skip",
                showlegend=False,
            ))

        # --- Invisible dummy trace for the edge-colour legend (colorbar) ---
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                size=0.001,
                color=[c_min, c_max],
                colorscale=edge_colorscale,
                colorbar=dict(
                    title=dict(text=edge_color_metric, font=dict(size=11)),
                    thickness=12,
                    len=0.5,
                    x=1.02,
                    y=0.75,
                    yanchor="middle",
                ),
                showscale=True,
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

    # --- Node traces (one per category for legend) ---
    # Compute node sizes based on out-strength
    strengths = dict(G.out_degree(weight="weight"))
    max_str = max(strengths.values()) if strengths and max(strengths.values()) > 0 else 1
    min_str = min(strengths.values()) if strengths else 0
    str_range = max_str - min_str if max_str > min_str else 1

    # Group nodes by category
    cat_nodes = {}
    for node in G.nodes():
        cat = node_categories.get(node, "Other Services")
        cat_nodes.setdefault(cat, []).append(node)

    for cat, nodes in sorted(cat_nodes.items()):
        x_vals, y_vals, sizes, texts, hover_texts = [], [], [], [], []
        for node in nodes:
            if node in pos:
                x, y = pos[node]
                x_vals.append(x)
                y_vals.append(y)
                s = strengths.get(node, 0)
                norm_s = (s - min_str) / str_range
                size = node_size_range[0] + norm_s * (node_size_range[1] - node_size_range[0])
                sizes.append(size)
                # Short label
                label = str(node)[:20]
                texts.append(label if size > 15 else "")
                # Hover text
                in_deg = G.in_degree(node)
                out_deg = G.out_degree(node)
                hover_texts.append(
                    f"<b>{node}</b><br>"
                    f"Category: {cat}<br>"
                    f"Out-strength: {s:,.0f}<br>"
                    f"In-degree: {in_deg}<br>"
                    f"Out-degree: {out_deg}"
                )

        color = category_colors.get(cat, "#999999")
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=color,
                line=dict(width=1, color="white"),
                opacity=0.85,
            ),
            text=texts,
            textposition="top center",
            textfont=dict(size=8),
            hovertext=hover_texts,
            hoverinfo="text",
            name=cat,
            legendgroup=cat,
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=80),
        height=650,
        hovermode="closest",
    )

    return fig
