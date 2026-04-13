"""Node detail panel — shows centrality and connections for a selected node."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_node_selector(G, node_features_df):
    """Render a node selector and detail panel.

    Args:
        G: nx.DiGraph for current quarter.
        node_features_df: DataFrame of node features indexed by node name.

    Returns:
        Selected node name or None.
    """
    nodes = sorted(G.nodes())
    # Sort by out-strength for more useful default ordering
    strengths = dict(G.out_degree(weight="weight"))
    nodes_sorted = sorted(nodes, key=lambda n: strengths.get(n, 0), reverse=True)

    selected = st.selectbox(
        "Inspect Industry",
        options=["(none)"] + nodes_sorted,
        key="node_selector",
    )

    if selected == "(none)":
        return None
    return selected


def render_node_details(G, node, node_features_df, all_node_features, quarters):
    """Render detailed information about a selected node.

    Args:
        G: nx.DiGraph for current quarter.
        node: Node name string.
        node_features_df: Current quarter's node features DataFrame.
        all_node_features: Dict[quarter, DataFrame] — all quarters' node features.
        quarters: Sorted list of quarter strings.
    """
    if node not in G:
        st.warning(f"Node '{node}' not found in current quarter's graph.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{node}**")

        # Centrality metrics
        if node in node_features_df.index:
            features = node_features_df.loc[node]
            metrics_df = pd.DataFrame({
                "Metric": [
                    "Betweenness Centrality",
                    "Eigenvector Centrality",
                    "In-Degree Centrality",
                    "Out-Degree Centrality",
                    "In-Strength",
                    "Out-Strength",
                    "Clustering Coefficient",
                ],
                "Value": [
                    f"{features.get('betweenness_centrality', 0):.4f}",
                    f"{features.get('eigenvector_centrality', 0):.4f}",
                    f"{features.get('in_degree_centrality', 0):.4f}",
                    f"{features.get('out_degree_centrality', 0):.4f}",
                    f"{features.get('in_strength', 0):,.0f}",
                    f"{features.get('out_strength', 0):,.0f}",
                    f"{features.get('clustering_coefficient', 0):.4f}",
                ],
            })
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    with col2:
        # Top connections
        st.markdown("**Top Outgoing Connections**")
        out_edges = [(v, d.get("weight", 0)) for _, v, d in G.out_edges(node, data=True)]
        out_edges.sort(key=lambda x: x[1], reverse=True)
        if out_edges:
            top_out = pd.DataFrame(out_edges[:5], columns=["To Industry", "Volume"])
            top_out["Volume"] = top_out["Volume"].apply(lambda x: f"{x:,.0f}")
            st.dataframe(top_out, hide_index=True, use_container_width=True)

        st.markdown("**Top Incoming Connections**")
        in_edges = [(u, d.get("weight", 0)) for u, _, d in G.in_edges(node, data=True)]
        in_edges.sort(key=lambda x: x[1], reverse=True)
        if in_edges:
            top_in = pd.DataFrame(in_edges[:5], columns=["From Industry", "Volume"])
            top_in["Volume"] = top_in["Volume"].apply(lambda x: f"{x:,.0f}")
            st.dataframe(top_in, hide_index=True, use_container_width=True)

    # Strength over time sparkline
    if all_node_features:
        out_strengths = []
        for q in quarters:
            nf = all_node_features.get(q)
            if nf is not None and node in nf.index:
                out_strengths.append(nf.loc[node, "out_strength"])
            else:
                out_strengths.append(0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=quarters, y=out_strengths,
            mode="lines+markers",
            name="Out-Strength",
            line=dict(color="#e41a1c"),
        ))
        fig.update_layout(
            title=f"Out-Strength Over Time: {node[:40]}",
            height=250,
            margin=dict(l=50, r=20, t=40, b=40),
            xaxis=dict(tickangle=45),
        )
        st.plotly_chart(fig, use_container_width=True)
