"""Quarter/year time slider component."""

import streamlit as st


def render_time_slider(quarters):
    """Render a quarter selection slider.

    Args:
        quarters: Sorted list of quarter strings (e.g., ['2017-Q1', ...]).

    Returns:
        Selected quarter string.
    """
    if not quarters:
        st.warning("No quarters available.")
        return None

    selected = st.select_slider(
        "Select Quarter",
        options=quarters,
        value=quarters[-1],
        key="quarter_slider",
    )
    return selected


def render_year_selector(quarters):
    """Render a year-based selector showing one representative quarter per year.

    Args:
        quarters: Sorted list of quarter strings.

    Returns:
        List of quarter strings for the selected year.
    """
    years = sorted(set(q.split("-")[0] for q in quarters))
    selected_year = st.select_slider(
        "Select Year",
        options=years,
        value=years[-1],
        key="year_slider",
    )
    return [q for q in quarters if q.startswith(selected_year)]
