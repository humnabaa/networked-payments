"""ONS live data loader — streams data directly from the ONS website into memory.

No files are saved to disk. Data is fetched fresh on each explicit load request
and held in Streamlit session state for the current session.
"""

import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# ONS dataset — only the 2019-2025 edition is currently published
ONS_URL = (
    "https://www.ons.gov.uk/file?uri=/economy/economicoutputandproductivity"
    "/output/datasets/industrytoindustrypaymentflowsukexperimentaldataandinsights"
    "/2019to2025/onsindustryflowssic2.xlsx"
)
ONS_SHEET = "SIC2_2019_2025"
ONS_SKIPROWS = 8  # 8 metadata rows before the column header row


def _fetch_ons_bytes() -> bytes | None:
    """Download the ONS Excel file and return raw bytes, or None on failure."""
    try:
        resp = requests.get(ONS_URL, timeout=120, stream=True)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        st.error(f"Failed to connect to ONS: {e}")
        return None


def _parse_ons_bytes(raw: bytes) -> pd.DataFrame | None:
    """Parse raw Excel bytes into a clean DataFrame with canonical columns.

    The ONS sheet has:
        Payer (2-digit SIC) | Payee (2-digit SIC) | Date | Value (£) | Number of transactions

    Returns DataFrame with columns: source, target, value, date, n_transactions
    """
    try:
        df = pd.read_excel(BytesIO(raw), sheet_name=ONS_SHEET, skiprows=ONS_SKIPROWS)

        rename_map = {
            "Payer (2-digit SIC)": "source",
            "Payee (2-digit SIC)": "target",
            "Date": "date",
            "Value (\xa3)": "value",          # may have encoding variant
            "Value (£)": "value",
            "Number of transactions": "n_transactions",
        }
        df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

        required = {"source", "target", "value", "date"}
        missing = required - set(df.columns)
        if missing:
            st.error(f"Could not find columns: {missing}. Columns present: {list(df.columns)}")
            return None

        # Drop suppressed / missing values
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df = df[df["value"] > 0]

        # Normalise SIC codes to strings
        df["source"] = df["source"].astype(str).str.strip().str.zfill(2)
        df["target"] = df["target"].astype(str).str.strip().str.zfill(2)

        return df.reset_index(drop=True)

    except Exception as e:
        st.error(f"Failed to parse ONS data: {e}")
        return None


def load_ons_data_live() -> pd.DataFrame | None:
    """Download and parse ONS data into memory. Stores result in session state.

    Returns the parsed DataFrame, or None if download/parse fails.
    """
    raw = _fetch_ons_bytes()
    if raw is None:
        return None

    df = _parse_ons_bytes(raw)
    if df is not None:
        st.session_state["ons_df"] = df
        st.session_state["ons_loaded"] = True
        size_mb = len(raw) / 1024 / 1024
        st.session_state["ons_info"] = (
            f"ONS 2019-2025 | {len(df):,} records | {size_mb:.1f} MB"
        )
    return df


def render_ons_loader() -> pd.DataFrame | None:
    """Render the ONS live loader panel in the sidebar.

    Always shows a Load / Reload button so users can refresh data at any time.
    Returns the loaded DataFrame (from session state or fresh download),
    or None if no data has been loaded yet.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ONS Live Data")
    st.sidebar.caption(
        "Loads industry-to-industry payment flows directly from the "
        "ONS website into memory."
    )

    already_loaded = st.session_state.get("ons_loaded", False)
    button_label = "Reload from ONS" if already_loaded else "Load from ONS (2019-2025)"

    if st.sidebar.button(button_label, key="ons_load_btn"):
        with st.spinner("Fetching ONS data (~33 MB)..."):
            df = load_ons_data_live()
        if df is not None:
            st.sidebar.success(
                f"Loaded {len(df):,} payment records into memory."
            )
            st.rerun()

    if already_loaded:
        info = st.session_state.get("ons_info", "ONS data loaded")
        st.sidebar.info(info)
        return st.session_state.get("ons_df")

    return None
