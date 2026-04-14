# Network Structure in UK Payment Flows

An open-source implementation of the research paper *"Network Structure in UK Payment Flows: Evidence on Economic Interdependencies and Implications for Real-Time Measurement"* (Humnabadkar, 2026).

This project analyses inter-industry payment flows published by the UK Office for National Statistics, constructs directed weighted network graphs, extracts graph-theoretic features, and evaluates whether network structure improves forecasting of payment flow dynamics. 

Try the app here : https://networked-payments-visualizer.streamlit.app/

## Project details

1. **Loads ONS payment data** — reads the experimental "Industry to Industry Payment Flows" Excel dataset, aggregates monthly records into quarterly bilateral flows across 88 industry sectors.
2. **Builds network graphs** — constructs directed weighted graphs per quarter where nodes are industries and edge weights are payment volumes.
3. **Extracts graph-theoretic features** — computes centrality measures (betweenness, eigenvector, degree), clustering coefficients, network density, average path length, and 2-hop connectivity.
4. **Trains forecasting models** — compares Traditional (lagged growth + seasonality), Network-only, and Combined specifications using Random Forest and Gradient Boosting with expanding-window cross-validation.
5. **Evaluates results** — produces R², RMSE, MAE metrics with bootstrap confidence intervals and Diebold-Mariano tests for statistical significance.
6. **Interactive dashboard** — a Streamlit app for exploring the network visually, with multiple graph views, edge colouring by network metrics, and temporal navigation.

## Project Structure

```
.
├── config/
│   └── settings.yaml          # Column mappings, model hyperparameters, period definitions
├── data/
│   └── raw/                   # Place your ONS Excel file here
├── outputs/
│   ├── figures/               # Generated figures (after pipeline run)
│   └── tables/                # Generated CSV tables (after pipeline run)
├── src/
│   ├── utils.py               # SIC code mappings, industry categorisation helpers
│   ├── data_loader.py         # ONS Excel/CSV loader with monthly-to-quarterly aggregation
│   ├── graph_builder.py       # Adjacency matrices, row-normalisation, 2-hop connectivity
│   ├── feature_extractor.py   # Node, edge, and network-level feature extraction
│   ├── target_builder.py      # Growth rate computation, lagged features, fixed effects
│   ├── model_trainer.py       # Expanding-window CV with RF and GBM
│   ├── evaluator.py           # Metrics, bootstrap CIs, Diebold-Mariano test
│   └── table_generator.py     # Publication-ready tables (Tables 1–4 from the paper)
├── viz/
│   ├── app.py                 # Streamlit dashboard entry point
│   └── components/
│       ├── network_graph.py   # Plotly network rendering with metric-based edge colouring
│       ├── metrics_panel.py   # Sidebar metrics and evolution charts
│       ├── node_details.py    # Industry inspector panel
│       └── time_slider.py     # Quarter selection slider
├── run_pipeline.py            # CLI entry point for the full analysis pipeline
├── pyproject.toml             # Project metadata and dependencies (for uv)
└── requirements.txt           # Dependencies (for pip)
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

**With uv (recommended):**

```bash
[git clone https://github.com/<your-username>/networked-payments.git](https://github.com/humnabaa/networked-payments.git)
cd networked-payments
uv sync
```

**With pip:**

```bash
[git clone https://github.com/<your-username>/networked-payments.git](https://github.com/humnabaa/networked-payments.git)
cd networked-payments
pip install -r requirements.txt
```

### Data

Download the ONS "Industry to Industry Payment Flows" dataset:

1. Go to the [ONS dataset page](https://www.ons.gov.uk/economy/economicoutputandproductivity/output/datasets/industrytoindustrypaymentflowsukexperimentaldataandinsights).
2. Download the SIC2 Excel file.
3. Place it in `data/raw/`.

## Usage

### Interactive Dashboard

```bash
# With uv
uv run streamlit run viz/app.py

# With pip
streamlit run viz/app.py
```

The dashboard will automatically detect the Excel file in `data/raw/` and load it. You can:

- **Navigate time** — use the quarter slider to move through 2019-Q1 to 2025-Q4.
- **Switch graph structures** — choose from Directed Weighted, Bipartite (Goods vs Services), Temporal Difference, Backbone, or Undirected views.
- **Colour edges by metrics** — select from 9 network metrics (betweenness centrality, in/out strength, eigenvector centrality, etc.) with 7 colour scales.
- **Inspect industries** — click on nodes to see centrality rankings, top connections, and strength over time.
- **Monitor network evolution** — track density, clustering, and path length trends across quarters.

### Full Analysis Pipeline

```bash
# With uv
uv run python run_pipeline.py \
    --data "data/raw/onsindustryflowssic2 (1).xlsx" \
    --config config/settings.yaml \
    --output outputs

# With pip
python run_pipeline.py \
    --data "data/raw/onsindustryflowssic2 (1).xlsx" \
    --config config/settings.yaml \
    --output outputs
```

This runs all 7 stages: data loading, graph construction, feature extraction, target building, model training, evaluation, and table generation.

> **Performance tip:** Model training with default hyperparameters (200 trees, expanding window CV over ~147K observations) is compute-intensive. For faster runs, reduce `n_estimators` in `config/settings.yaml`:
>
> ```yaml
> model:
>   rf_params:
>     n_estimators: 50
>     max_depth: 6
>   gbm_params:
>     n_estimators: 50
>     max_depth: 4
> ```

## Key Results (from the paper)

| Finding | Detail |
|---------|--------|
| Forecasting improvement | Combined model achieves R² = 0.412 (+8.8 pp over traditional) |
| Crisis resilience | Network contribution doubles during COVID-19 (+13.8 pp vs +6.0 pp in stable periods) |
| Structurally central industries | Financial Services, Wholesale Trade, Professional Services |
| Network densification | Density grew 12.5% from 2019 to 2024 |

## Configuration

All settings are in `config/settings.yaml`:

- **`schema`** — column name mappings for your data file
- **`industry_categories`** — SIC code ranges and colours for visualisation groupings
- **`model`** — Random Forest and Gradient Boosting hyperparameters
- **`periods`** — date boundaries for pre-pandemic, pandemic, and recovery analysis
- **`visualization`** — max edges, node sizes, layout parameters

## Data Source

This project uses the ONS experimental [Industry to Industry Payment Flows](https://www.ons.gov.uk/economy/economicoutputandproductivity/output/datasets/industrytoindustrypaymentflowsukexperimentaldataandinsights) dataset. The data comprises anonymised, aggregated Bacs and Faster Payment Service transactions between UK organisations, classified by 2-digit SIC codes.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{humnabadkar2026network,
  title={Network Structure in UK Payment Flows: Evidence on Economic Interdependencies and Implications for Real-Time Measurement},
  author={Humnabadkar, Aditya},
  journal={arXiv preprint arXiv:2604.02068},
  year={2026}
}
```

## Licence

This project is open source. See [LICENSE]([LICENSE](https://github.com/humnabaa/networked-payments?tab=MIT-1-ov-file)) for details.
