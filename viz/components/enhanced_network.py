"""Enhanced Network Graph Visualization with Advanced Interaction Features."""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import pandas as pd


class InteractivePaymentNetworkVisualizer:
    """Enhanced network visualization with path tracing and relationship analysis."""
    
    def __init__(self, G, pos, node_categories, category_colors):
        """Initialize visualizer.
        
        Args:
            G: NetworkX DiGraph
            pos: Dict of node positions
            node_categories: Dict mapping node -> category
            category_colors: Dict mapping category -> color
        """
        self.G = G
        self.pos = pos
        self.node_categories = node_categories
        self.category_colors = category_colors
    
    def render_relationship_analyzer(self):
        """Render tool to analyze relationships between two entities."""
        st.subheader("Relationship Analyzer")
        
        nodes = sorted(self.G.nodes())
        col1, col2 = st.columns(2)
        
        with col1:
            source_node = st.selectbox(
                "Source Industry",
                options=nodes,
                key="rel_source"
            )
        
        with col2:
            target_node = st.selectbox(
                "Target Industry",
                options=nodes,
                key="rel_target",
                index=1 if len(nodes) > 1 else 0
            )
        
        if source_node and target_node:
            self._analyze_pair(source_node, target_node)
    
    def _analyze_pair(self, source, target):
        """Analyze relationship between two nodes."""
        st.markdown(f"#### {source} → {target}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Direct edge info
            if self.G.has_edge(source, target):
                weight = self.G[source][target].get("weight", 0)
                st.metric("Direct Payment Volume", f"£{weight:,.0f}")
                
                # Additional edge attributes
                for key, val in self.G[source][target].items():
                    if key != "weight" and isinstance(val, (int, float)):
                        st.metric(f"Edge: {key}", f"{val:.3f}")
            else:
                st.warning("No direct payment flow between these industries")
        
        with col2:
            st.metric("In Same Category?", 
                     "Yes" if self.node_categories.get(source) == self.node_categories.get(target) else "No")
        
        # Common neighbors
        st.markdown("##### Common Trading Partners")
        
        source_suppliers = set(self.G.predecessors(source))
        source_customers = set(self.G.successors(source))
        
        target_suppliers = set(self.G.predecessors(target))
        target_customers = set(self.G.successors(target))
        
        common_suppliers = source_suppliers & target_suppliers
        common_customers = source_customers & target_customers
        
        col1, col2 = st.columns(2)
        with col1:
            if common_suppliers:
                st.write(f"**Both buy from** ({len(common_suppliers)})")
                for ind in sorted(common_suppliers):
                    st.caption(f"  • {ind}")
            else:
                st.caption("No common suppliers")
        
        with col2:
            if common_customers:
                st.write(f"**Both sell to** ({len(common_customers)})")
                for ind in sorted(common_customers):
                    st.caption(f"  • {ind}")
            else:
                st.caption("No common customers")
        
        # Shortest paths
        st.markdown("##### Network Paths")
        try:
            if nx.has_path(self.G.to_undirected(), source, target):
                path_len = nx.shortest_path_length(self.G.to_undirected(), source, target)
                st.metric("Shortest Path Length", path_len)
                
                # Try to show shortest path
                try:
                    shortest = nx.shortest_path(self.G.to_undirected(), source, target)
                    st.caption(f"Path: {' → '.join(shortest)}")
                except:
                    pass
            else:
                st.warning("No path exists between these industries (in undirected graph)")
        except Exception as e:
            st.warning(f"Could not compute path: {e}")
    
    def render_subnetwork_extractor(self):
        """Extract and visualize subnetwork around selected nodes."""
        st.subheader("Subnetwork Explorer")
        
        nodes = sorted(self.G.nodes())
        selected_nodes = st.multiselect(
            "Select nodes to extract subnetwork",
            options=nodes,
            default=nodes[:3] if len(nodes) >= 3 else nodes,
            key="subnetwork_nodes"
        )
        
        if selected_nodes:
            hop_distance = st.slider(
                "Hops from selected nodes",
                min_value=0,
                max_value=3,
                value=1,
                key="hop_distance"
            )
            
            # Extract subnetwork
            subgraph_nodes = set(selected_nodes)
            
            for _ in range(hop_distance):
                new_nodes = set()
                for n in subgraph_nodes:
                    new_nodes.update(self.G.predecessors(n))
                    new_nodes.update(self.G.successors(n))
                subgraph_nodes.update(new_nodes)
            
            subgraph = self.G.subgraph(subgraph_nodes).copy()
            
            st.write(f"**Subnetwork:** {len(subgraph.nodes())} nodes, {len(subgraph.edges())} edges")
            
            # Visualize subnetwork
            fig = self._create_subnetwork_figure(subgraph, selected_nodes)
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_subnetwork_figure(self, G, highlighted_nodes):
        """Create Plotly figure for subnetwork."""
        
        # Compute positions for subgraph
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        edge_x, edge_y = [], []
        edge_colors = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            
            # Color by weight
            weight = edge[2].get("weight", 1)
            edge_colors.append(np.log1p(weight))
        
        fig = go.Figure()
        
        # Add edges
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=1, color='rgba(125,125,125,0.3)'),
                hoverinfo='none',
                showlegend=False,
                name='edges'
            ))
        
        # Add nodes
        node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Color and size
            if node in highlighted_nodes:
                node_color.append("#FF6B6B")
                node_size.append(20)
            else:
                category = self.node_categories.get(node, "Other Services")
                node_color.append(self.category_colors.get(category, "#999999"))
                node_size.append(12)
            
            node_text.append(f"<b>{node}</b>")
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=node_size, color=node_color, line=dict(width=2, color='white')),
            text=[n[:10] for n in G.nodes()],  # Truncate names
            textposition="top center",
            textfont=dict(size=9),
            hovertext=node_text,
            hoverinfo='text',
            showlegend=False,
        ))
        
        fig.update_layout(
            title="Subnetwork View",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            plot_bgcolor='#f8f9fa',
        )
        
        return fig
    
    def render_centrality_hubs(self):
        """Analyze and visualize network hubs."""
        st.subheader("Network Hubs Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        # Compute various centrality measures
        out_degree = dict(self.G.out_degree(weight="weight"))
        in_degree = dict(self.G.in_degree(weight="weight"))
        
        try:
            betweenness = nx.betweenness_centrality(self.G, weight="weight")
        except:
            betweenness = {}
        
        with col1:
            st.write("**Top Payers (Out-Strength)**")
            top_payers = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5]
            for ind, val in top_payers:
                st.caption(f"{ind}: £{val:,.0f}")
        
        with col2:
            st.write("**Top Recipients (In-Strength)**")
            top_recipients = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5]
            for ind, val in top_recipients:
                st.caption(f"{ind}: £{val:,.0f}")
        
        with col3:
            st.write("**Bridge Industries (Betweenness)**")
            if betweenness:
                top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
                for ind, val in top_bridges:
                    st.caption(f"{ind}: {val:.3f}")
            else:
                st.caption("Computing...")


def render_enhanced_network(G, pos, node_categories, category_colors, node_features=None):
    """Main function to render enhanced network visualization.
    
    Args:
        G: NetworkX DiGraph
        pos: Node positions dict
        node_categories: Mapping of node -> category
        category_colors: Mapping of category -> color
        node_features: Optional dict of node-level metrics
    """
    st.markdown("---")
    st.markdown("## Advanced Network Analysis")
    
    # Create visualizer instance
    viz = InteractivePaymentNetworkVisualizer(G, pos, node_categories, category_colors)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Relationship Analysis",
        "Subnetwork Explorer",
        "Hub Analysis",
        "Network Stats"
    ])
    
    with tab1:
        viz.render_relationship_analyzer()
    
    with tab2:
        viz.render_subnetwork_extractor()
    
    with tab3:
        viz.render_centrality_hubs()
    
    with tab4:
        render_network_statistics(G)


def render_network_statistics(G):
    """Render comprehensive network statistics."""
    st.subheader("Network Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nodes", len(G.nodes()))
    with col2:
        st.metric("Edges", len(G.edges()))
    with col3:
        total_flow = sum(d.get('weight', 0) for _, _, d in G.edges(data=True))
        st.metric("Total Payment Flow", f"£{total_flow:,.0f}")
    with col4:
        density = nx.density(G)
        st.metric("Network Density", f"{density:.3f}")
    
    # Distribution statistics
    st.markdown("### Degree Distribution")
    
    out_degrees = [d for _, d in G.out_degree(weight="weight")]
    in_degrees = [d for _, d in G.in_degree(weight="weight")]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Out-Degree (Paying)**")
        st.caption(f"Mean: {np.mean(out_degrees):,.0f}")
        st.caption(f"Median: {np.median(out_degrees):,.0f}")
        st.caption(f"Max: {np.max(out_degrees):,.0f}")
    
    with col2:
        st.write("**In-Degree (Receiving)**")
        st.caption(f"Mean: {np.mean(in_degrees):,.0f}")
        st.caption(f"Median: {np.median(in_degrees):,.0f}")
        st.caption(f"Max: {np.max(in_degrees):,.0f}")
    
    with col3:
        st.write("**Flow Statistics**")
        weights = [d.get('weight', 0) for _, _, d in G.edges(data=True)]
        st.caption(f"Mean Flow: £{np.mean(weights):,.0f}")
        st.caption(f"Median Flow: £{np.median(weights):,.0f}")
        st.caption(f"Total Flow: £{np.sum(weights):,.0f}")
