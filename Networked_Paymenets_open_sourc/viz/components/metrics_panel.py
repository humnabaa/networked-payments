"""Sidebar metrics display panel."""

import streamlit as st
import plotly.graph_objects as go


def render_metrics_sidebar(network_features, selected_quarter, quarters):
    """Render network metrics in the Streamlit sidebar.

    Args:
        network_features: Dict[quarter_str, dict of metrics].
        selected_quarter: Currently selected quarter string.
        quarters: Sorted list of all quarter strings.
    """
    current = network_features.get(selected_quarter, {})
    q_idx = quarters.index(selected_quarter) if selected_quarter in quarters else 0
    prev_quarter = quarters[q_idx - 1] if q_idx > 0 else None
    prev = network_features.get(prev_quarter, {}) if prev_quarter else {}

    st.sidebar.markdown("### Network Metrics")

    # Density
    density = current.get("density", 0)
    prev_density = prev.get("density")
    delta_d = f"{(density - prev_density):.3f}" if prev_density is not None else None
    st.sidebar.metric("Density", f"{density:.3f}", delta=delta_d)

    # Edges
    edges = current.get("num_edges", 0)
    prev_edges = prev.get("num_edges")
    delta_e = f"{edges - prev_edges}" if prev_edges is not None else None
    st.sidebar.metric("Edges", f"{edges:,}", delta=delta_e)

    # Avg Path Length
    apl = current.get("average_path_length", 0)
    prev_apl = prev.get("average_path_length")
    delta_apl = f"{(apl - prev_apl):.2f}" if prev_apl is not None else None
    st.sidebar.metric("Avg Path Length", f"{apl:.2f}", delta=delta_apl, delta_color="inverse")

    # Clustering
    clust = current.get("average_clustering", 0)
    prev_clust = prev.get("average_clustering")
    delta_c = f"{(clust - prev_clust):.3f}" if prev_clust is not None else None
    st.sidebar.metric("Avg Clustering", f"{clust:.3f}", delta=delta_c)


def render_metrics_timeseries(network_features, quarters):
    """Render a line chart of network metrics over time.

    Args:
        network_features: Dict[quarter_str, dict of metrics].
        quarters: Sorted list of quarter strings.
    """
    densities = [network_features[q].get("density", 0) for q in quarters]
    path_lengths = [network_features[q].get("average_path_length", 0) for q in quarters]
    clustering = [network_features[q].get("average_clustering", 0) for q in quarters]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=quarters, y=densities, name="Density", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=quarters, y=path_lengths, name="Avg Path Length", mode="lines+markers", yaxis="y2"))
    fig.add_trace(go.Scatter(x=quarters, y=clustering, name="Avg Clustering", mode="lines+markers"))

    fig.update_layout(
        title="Network Metrics Over Time",
        xaxis=dict(title="Quarter", tickangle=45),
        yaxis=dict(title="Density / Clustering", side="left"),
        yaxis2=dict(title="Avg Path Length", side="right", overlaying="y"),
        height=300,
        margin=dict(l=50, r=50, t=40, b=60),
        legend=dict(orientation="h", y=-0.3),
        hovermode="x unified",
    )

    return fig
