"""Streamlit interactive visualization for payment network analysis.

Run with: streamlit run viz/app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import networkx as nx

from src.utils import load_config, get_industry_category, get_category_color
from src.data_loader import load_payment_data
from src.graph_builder import build_quarterly_graphs, get_node_list, build_adjacency_matrix
from src.feature_extractor import extract_node_features, extract_network_features
from viz.components.network_graph import (
    compute_stable_layout,
    create_network_figure,
    EXCLUDED_INDUSTRIES,
    EDGE_COLOR_METRICS,
    EDGE_COLORSCALES,
)
from viz.components.metrics_panel import render_metrics_sidebar, render_metrics_timeseries
from viz.components.time_slider import render_time_slider
from viz.components.node_details import render_node_selector, render_node_details

st.set_page_config(
    page_title="Payment Network Visualizer",
    page_icon="",
    layout="wide",
)

# ─── Graph structure transformations ────────────────────────────────────────

GRAPH_STRUCTURES = {
    "Directed Weighted (default)": "directed",
    "Bipartite (Goods vs Services)": "bipartite",
    "Temporal Difference": "temporal_diff",
    "Backbone (significant edges)": "backbone",
    "Undirected (symmetric)": "undirected",
}

# SIC ranges that are broadly "goods-producing"
_GOODS_SICS = set(range(1, 40)) | {41, 42, 43, 45, 46, 47}


def _is_goods_industry(node_name, sic_lookup):
    """Return True if the industry is goods-producing."""
    sic = sic_lookup.get(node_name)
    return sic is not None and sic in _GOODS_SICS


def transform_graph(G, mode, graphs, selected_quarter, quarters, sic_lookup):
    """Return a transformed copy of G according to *mode*.

    Args:
        G: Original nx.DiGraph for the selected quarter.
        mode: One of the values in GRAPH_STRUCTURES.
        graphs: Full dict of quarterly graphs (for temporal diff).
        selected_quarter: Currently selected quarter string.
        quarters: Sorted quarter list.
        sic_lookup: Dict mapping industry name -> SIC int.

    Returns:
        Transformed nx.DiGraph (or nx.Graph for undirected modes).
    """
    if mode == "directed":
        return G

    if mode == "bipartite":
        # Create a bipartite projection: keep only edges that cross the
        # goods / services divide.  Nodes retain their original names
        # but get a 'bipartite' attribute (0 = Goods, 1 = Services).
        B = G.copy()
        for n in B.nodes():
            B.nodes[n]["bipartite"] = 0 if _is_goods_industry(n, sic_lookup) else 1
        cross_edges = [
            (u, v)
            for u, v in B.edges()
            if B.nodes[u]["bipartite"] != B.nodes[v]["bipartite"]
        ]
        keep = set()
        for u, v in cross_edges:
            keep.add(u)
            keep.add(v)
        remove = [n for n in B.nodes() if n not in keep]
        B.remove_nodes_from(remove)
        intra_edges = [
            (u, v)
            for u, v in list(B.edges())
            if B.nodes[u]["bipartite"] == B.nodes[v]["bipartite"]
        ]
        B.remove_edges_from(intra_edges)
        return B

    if mode == "temporal_diff":
        # Edge weight = change from previous quarter.
        q_idx = quarters.index(selected_quarter) if selected_quarter in quarters else 0
        if q_idx == 0:
            return G  # No previous quarter
        prev_q = quarters[q_idx - 1]
        G_prev = graphs.get(prev_q)
        if G_prev is None:
            return G
        D = nx.DiGraph()
        D.add_nodes_from(G.nodes())
        for u, v, data in G.edges(data=True):
            w_now = data.get("weight", 0)
            w_prev = G_prev[u][v]["weight"] if G_prev.has_edge(u, v) else 0
            diff = w_now - w_prev
            if diff != 0:
                D.add_edge(u, v, weight=abs(diff), raw_diff=diff)
        return D

    if mode == "backbone":
        # Keep only edges whose weight exceeds the median edge weight,
        # i.e. the statistically "significant" backbone of the network.
        weights = [d.get("weight", 0) for _, _, d in G.edges(data=True)]
        if not weights:
            return G
        import numpy as _np
        threshold = _np.percentile(weights, 75)  # top quartile
        B = nx.DiGraph()
        B.add_nodes_from(G.nodes())
        for u, v, d in G.edges(data=True):
            if d.get("weight", 0) >= threshold:
                B.add_edge(u, v, **d)
        return B

    if mode == "undirected":
        U_nx = G.to_undirected()
        # Sum weights for reciprocal edges
        U = nx.Graph()
        U.add_nodes_from(G.nodes())
        for u, v, d in U_nx.edges(data=True):
            w = d.get("weight", 0)
            if G.has_edge(v, u):
                w += G[v][u].get("weight", 0)
            U.add_edge(u, v, weight=w)
        return U

    return G


# ─── Caching helpers ────────────────────────────────────────────────────────

@st.cache_data
def load_and_process_data(data_path, config_path):
    """Load data and build graphs (cached)."""
    config = load_config(config_path)
    df = load_payment_data(data_path, config)
    graphs = build_quarterly_graphs(df)
    node_list = get_node_list(graphs)
    return df, graphs, node_list, config


@st.cache_data
def compute_all_features(_graphs, _node_list):
    """Compute node and network features for all quarters (cached)."""
    node_features = {}
    network_features = {}
    for quarter in sorted(_graphs.keys()):
        G = _graphs[quarter]
        node_features[quarter] = extract_node_features(G)
        network_features[quarter] = extract_network_features(G)
    return node_features, network_features


@st.cache_data
def get_layout(_graphs, _node_list, seed, k):
    """Compute stable layout (cached)."""
    return compute_stable_layout(_graphs, _node_list, seed=seed, k=k)


def build_sic_lookup():
    """Build industry_name -> SIC code lookup."""
    from src.utils import SIC_INDUSTRY_NAMES
    return {name: code for code, name in SIC_INDUSTRY_NAMES.items()}


def build_category_maps(node_list, config):
    """Build node -> category and category -> color mappings."""
    from src.utils import SIC_INDUSTRY_NAMES
    name_to_sic = {name: code for code, name in SIC_INDUSTRY_NAMES.items()}

    node_categories = {}
    for node in [n for n in node_list if n not in EXCLUDED_INDUSTRIES]:
        sic = name_to_sic.get(node)
        if sic is not None:
            node_categories[node] = get_industry_category(sic, config)
        else:
            node_categories[node] = "Other Services"

    category_colors = {}
    for cat_name, cat_info in config.get("industry_categories", {}).items():
        category_colors[cat_name] = cat_info.get("color", "#999999")

    return node_categories, category_colors


# ─── Main application ──────────────────────────────────────────────────────

def main():
    st.title("UK Inter-Industry Payment Network")
    st.caption("Interactive visualization of network structure and evolution")

    # --- Sidebar: Data Source ---
    st.sidebar.title("Data Source")

    config_path = "config/settings.yaml"
    data_path = None

    # Check for bundled data first
    default_data = Path("data/raw")
    bundled_files = list(default_data.glob("*.xlsx")) + list(default_data.glob("*.csv"))

    if bundled_files:
        if len(bundled_files) == 1:
            data_path = str(bundled_files[0])
            st.sidebar.success(f"Using: {bundled_files[0].name}")
        else:
            selected_file = st.sidebar.selectbox(
                "Select dataset",
                options=[f.name for f in bundled_files],
                key="bundled_file",
            )
            data_path = str(default_data / selected_file)
    else:
        # Fall back to upload if no bundled data
        uploaded = st.sidebar.file_uploader(
            "Upload payment data",
            type=["csv", "xlsx", "xls"],
        )
        if uploaded:
            upload_path = f"data/raw/{uploaded.name}"
            Path(upload_path).parent.mkdir(parents=True, exist_ok=True)
            with open(upload_path, "wb") as f:
                f.write(uploaded.getvalue())
            data_path = upload_path
            st.session_state["data_path"] = data_path
        else:
            data_path = st.session_state.get("data_path")

    if data_path is None:
        st.info("No data found. Place an Excel or CSV file in `data/raw/` or upload one above.")
        return

    # --- Load and process data ---
    try:
        df, graphs, node_list, config = load_and_process_data(data_path, config_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    quarters = sorted(graphs.keys())

    # Filter out excluded industries (e.g. "Unknown or Unclassified")
    node_list = [n for n in node_list if n not in EXCLUDED_INDUSTRIES]
    for q in graphs:
        for excl in EXCLUDED_INDUSTRIES:
            if excl in graphs[q]:
                graphs[q].remove_node(excl)

    node_features, network_features = compute_all_features(graphs, node_list)

    viz_config = config.get("visualization", {})
    pos = get_layout(graphs, node_list, viz_config.get("layout_seed", 42), viz_config.get("layout_k", 0.3))
    node_categories, category_colors = build_category_maps(node_list, config)
    sic_lookup = build_sic_lookup()

    # --- Sidebar: Graph Structure ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Graph Structure")
    graph_mode_label = st.sidebar.selectbox(
        "View as",
        options=list(GRAPH_STRUCTURES.keys()),
        index=0,
        key="graph_structure",
        help=(
            "**Directed Weighted** — original payment flows.\n\n"
            "**Bipartite** — only cross-sector edges (Goods vs Services).\n\n"
            "**Temporal Difference** — edge weight = change from previous quarter.\n\n"
            "**Backbone** — top-quartile edges only (statistically significant).\n\n"
            "**Undirected** — symmetric view, reciprocal weights summed."
        ),
    )
    graph_mode = GRAPH_STRUCTURES[graph_mode_label]

    # --- Sidebar: Edge Colouring ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Edge Colouring")
    edge_metric = st.sidebar.selectbox(
        "Colour edges by",
        options=list(EDGE_COLOR_METRICS.keys()),
        index=0,
        key="edge_color_metric",
    )
    edge_cscale = st.sidebar.selectbox(
        "Colour scale",
        options=list(EDGE_COLORSCALES.keys()),
        index=0,
        key="edge_colorscale",
    )
    st.sidebar.caption(EDGE_COLOR_METRICS[edge_metric]["description"])

    # --- Sidebar: Metrics ---
    st.sidebar.markdown("---")

    # --- Time Slider ---
    selected_quarter = render_time_slider(quarters)
    if selected_quarter is None:
        return

    # Show metrics in sidebar
    render_metrics_sidebar(network_features, selected_quarter, quarters)

    # --- Transform graph ---
    G = graphs[selected_quarter]
    G_viz = transform_graph(G, graph_mode, graphs, selected_quarter, quarters, sic_lookup)

    # Get node features for colour metric look-ups
    nf_df = node_features.get(selected_quarter, pd.DataFrame())

    # --- Main: Network Graph ---
    title_suffix = f" [{graph_mode_label}]" if graph_mode != "directed" else ""
    fig = create_network_figure(
        G_viz,
        pos,
        node_categories,
        category_colors,
        title=f"Payment Network \u2014 {selected_quarter}{title_suffix}",
        max_edges=viz_config.get("max_edges_displayed", 300),
        node_size_range=tuple(viz_config.get("node_size_range", [8, 40])),
        edge_width_range=tuple(viz_config.get("edge_width_range", [0.3, 3.0])),
        edge_color_metric=edge_metric,
        edge_colorscale=EDGE_COLORSCALES[edge_cscale],
        node_features_df=nf_df,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Bottom panels ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Industry Details")
        selected_node = render_node_selector(G, nf_df)
        if selected_node:
            render_node_details(G, selected_node, nf_df, node_features, quarters)

    with col_right:
        st.subheader("Network Evolution")
        ts_fig = render_metrics_timeseries(network_features, quarters)
        st.plotly_chart(ts_fig, use_container_width=True)

    # --- Summary stats ---
    st.markdown("---")
    n_edges_viz = G_viz.number_of_edges()
    n_nodes_viz = G_viz.number_of_nodes()
    st.markdown(
        f"**Data summary:** {len(df):,} records | "
        f"{len(node_list)} industries | "
        f"{len(quarters)} quarters | "
        f"Showing {n_nodes_viz} nodes, {n_edges_viz} edges"
    )


if __name__ == "__main__":
    main()
