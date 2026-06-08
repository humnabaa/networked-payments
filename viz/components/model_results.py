"""ML Model Training Results Visualization Component."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path


def render_model_results_section(spec_results=None, eval_results=None):
    """Render ML model training results visualization section.
    
    Args:
        spec_results: Dict from model_trainer.train_all_specifications
        eval_results: Dict mapping spec_name -> metrics from evaluator
    """
    if not spec_results or not eval_results:
        st.info("⏳ Run the pipeline to generate model results (`python run_pipeline.py --generate-sample`)")
        return
    
    st.markdown("---")
    st.markdown("## ML Model Training Results")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Performance",
        "Learning Curves",
        "Feature Importance",
        "Diebold-Mariano Tests"
    ])
    
    with tab1:
        render_model_comparison(spec_results, eval_results)
    
    with tab2:
        render_learning_curves(spec_results)
    
    with tab3:
        render_feature_importance(spec_results)
    
    with tab4:
        render_diebold_mariano(spec_results, eval_results)


def render_model_comparison(spec_results, eval_results):
    """Render model performance comparison table and charts."""
    st.subheader("Model Performance Comparison")
    
    # Extract metrics for each specification
    specs = list(eval_results.keys())
    models = ["RF", "GBM"]
    
    metrics_data = []
    for spec in specs:
        for model in models:
            spec_key = f"{spec}_{model}"
            if spec_key in eval_results:
                metrics = eval_results[spec_key]
                metrics_data.append({
                    "Specification": spec,
                    "Model": model,
                    "R²": metrics.get("r2", 0),
                    "R² CI": f"({metrics.get('r2_ci', (0,0))[0]:.3f}, {metrics.get('r2_ci', (0,0))[1]:.3f})",
                    "RMSE": metrics.get("rmse", 0),
                    "MAE": metrics.get("mae", 0),
                })
    
    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        
        # Display as table
        st.dataframe(
            df_metrics,
            use_container_width=True,
            hide_index=True,
        )
        
        # Create comparison chart
        fig = go.Figure()
        
        for spec in specs:
            for model in models:
                spec_key = f"{spec}_{model}"
                if spec_key in eval_results:
                    r2 = eval_results[spec_key].get("r2", 0)
                    fig.add_trace(go.Bar(
                        name=f"{spec} ({model})",
                        x=[spec],
                        y=[r2],
                        text=f"{r2:.3f}",
                        textposition="outside",
                    ))
        
        fig.update_layout(
            title="R² Score by Model Specification",
            yaxis_title="R² Score",
            xaxis_title="Model Specification",
            barmode="group",
            height=400,
            showlegend=True,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No evaluation results available")


def render_learning_curves(spec_results):
    """Render learning curves showing model performance over folds."""
    st.subheader("Learning Curves (Expanding Window)")
    
    specs = list(spec_results.keys())
    models = ["RF", "GBM"]
    
    fig = go.Figure()
    
    for spec in specs:
        for model in models:
            spec_key = f"{spec}_{model}"
            if spec_key in spec_results:
                cv_results = spec_results[spec_key].get("cv_results", {})
                fold_results = cv_results.get("fold_results", [])
                
                if fold_results:
                    df_folds = pd.DataFrame(fold_results)
                    
                    fig.add_trace(go.Scatter(
                        x=df_folds["period"].astype(str),
                        y=df_folds["r2"],
                        name=f"{spec} ({model})",
                        mode="lines+markers",
                        hovertemplate="<b>%{name}</b><br>Period: %{x}<br>R²: %{y:.3f}<extra></extra>",
                    ))
    
    if len(fig.data) > 0:
        fig.update_layout(
            title="Model R² Score Over Time (Expanding Window CV)",
            xaxis_title="Time Period (Fold)",
            yaxis_title="R² Score",
            height=400,
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No learning curve data available")


def render_feature_importance(spec_results):
    """Render feature importance from tree-based models."""
    st.subheader("Feature Importance (Tree-Based Models)")
    
    specs = list(spec_results.keys())
    models = ["RF", "GBM"]
    
    col1, col2 = st.columns(2)
    
    for i, spec in enumerate(specs[:2]):  # Show first 2 specs
        col = col1 if i % 2 == 0 else col2
        
        with col:
            for model in models:
                spec_key = f"{spec}_{model}"
                if spec_key in spec_results:
                    feature_importance = spec_results[spec_key].get("feature_importance", {})
                    
                    if feature_importance:
                        # Get top 10 features
                        top_features = sorted(
                            feature_importance.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:10]
                        
                        features, importances = zip(*top_features)
                        
                        fig = go.Figure(data=[
                            go.Bar(y=list(features), x=list(importances), orientation='h')
                        ])
                        
                        fig.update_layout(
                            title=f"{spec} - {model} Top 10 Features",
                            xaxis_title="Importance",
                            yaxis_title="Feature",
                            height=300,
                            margin=dict(l=150),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        break  # Show only first model


def render_diebold_mariano(spec_results, eval_results):
    """Render Diebold-Mariano test results."""
    st.subheader("Diebold-Mariano Forecast Comparison Tests")
    
    dm_results = []
    
    for spec in eval_results.keys():
        if "dm_test" in eval_results[spec]:
            dm = eval_results[spec]["dm_test"]
            dm_results.append({
                "Comparison": spec,
                "DM Statistic": dm.get("dm_stat", 0),
                "P-Value": dm.get("p_value", 1),
                "Significant (alpha=0.05)": "Yes" if dm.get("p_value", 1) < 0.05 else "No",
            })
    
    if dm_results:
        df_dm = pd.DataFrame(dm_results)
        st.dataframe(df_dm, use_container_width=True, hide_index=True)
        
        st.caption(
            "DM Test: Compares forecast accuracy between models. "
            "Positive DM statistic indicates model 2 outperforms model 1."
        )
    else:
        st.info("No Diebold-Mariano test results available")


def load_model_results_from_files(output_dir="outputs"):
    """Load model results from output CSV files if they exist.
    
    Args:
        output_dir: Directory where results are saved
    
    Returns:
        Tuple of (spec_results, eval_results) or (None, None)
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return None, None
    
    spec_results = {}
    eval_results = {}
    
    # Try to load CSV files
    for csv_file in output_path.glob("*_results.csv"):
        try:
            df = pd.read_csv(csv_file)
            spec_name = csv_file.stem.replace("_results", "")
            spec_results[spec_name] = {"fold_results": df.to_dict('records')}
        except Exception as e:
            st.warning(f"Could not load {csv_file}: {e}")
    
    return spec_results if spec_results else None, eval_results if eval_results else None
