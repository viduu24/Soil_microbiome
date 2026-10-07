# src/data_source.py
"""
Shared data-loading layer used by every dashboard page. Each page needs
combined_long / family_wide / phylum_wide / alpha_diversity, and
previously read them directly from local processed/*.parquet files
(written by running wrangle.py against hardcoded file paths).

To let OTHER researchers use this dashboard on their own data (uploaded
directly in the browser via the "Upload Data" page, rather than editing
wrangle.py and running it from the command line), this module checks
Streamlit session_state FIRST for uploaded/processed data, and only
falls back to the local parquet files if nothing was uploaded this
session. This means:

  - The original workflow (edit wrangle.py, run it locally, launch the
    dashboard) keeps working completely unchanged.
  - A researcher who instead uses the "Upload Data" page gets the exact
    same downstream pages/analyses, with no code editing required.

Session state keys used (set by pages/0_Upload_Data.py):
  uploaded_combined_long, uploaded_family_wide, uploaded_phylum_wide,
  uploaded_alpha_diversity, uploaded_core_microbiome
"""

import streamlit as st
import pandas as pd
from pathlib import Path

_SESSION_KEYS = {
    "combined_long": "uploaded_combined_long",
    "family_wide": "uploaded_family_wide",
    "phylum_wide": "uploaded_phylum_wide",
    "alpha_diversity": "uploaded_alpha_diversity",
    "core_microbiome": "uploaded_core_microbiome",
}


def using_uploaded_data() -> bool:
    """True if a dataset was uploaded and processed via the Upload Data page this session."""
    return st.session_state.get(_SESSION_KEYS["combined_long"]) is not None


def data_available(processed_dir: Path) -> bool:
    """True if EITHER an uploaded dataset exists this session, OR the local combined_long.parquet exists."""
    return using_uploaded_data() or (processed_dir / "combined_long.parquet").exists()


def get_combined_long(processed_dir: Path) -> pd.DataFrame:
    key = _SESSION_KEYS["combined_long"]
    if st.session_state.get(key) is not None:
        return st.session_state[key]
    return pd.read_parquet(processed_dir / "combined_long.parquet")


def get_wide(processed_dir: Path, rank: str) -> pd.DataFrame:
    key = _SESSION_KEYS["family_wide"] if rank == "Family" else _SESSION_KEYS["phylum_wide"]
    if st.session_state.get(key) is not None:
        return st.session_state[key]
    fname = "family_wide.parquet" if rank == "Family" else "phylum_wide.parquet"
    return pd.read_parquet(processed_dir / fname)


def get_alpha_diversity(processed_dir: Path) -> pd.DataFrame:
    key = _SESSION_KEYS["alpha_diversity"]
    if st.session_state.get(key) is not None:
        return st.session_state[key]
    return pd.read_parquet(processed_dir / "alpha_diversity.parquet")


def get_core_microbiome(processed_dir: Path) -> pd.DataFrame:
    key = _SESSION_KEYS["core_microbiome"]
    if st.session_state.get(key) is not None:
        return st.session_state[key]
    return pd.read_parquet(processed_dir / "core_microbiome.parquet")


def store_uploaded_results(
    combined_long: pd.DataFrame,
    family_wide: pd.DataFrame,
    phylum_wide: pd.DataFrame,
    alpha_diversity: pd.DataFrame,
    core_microbiome: pd.DataFrame = None,
) -> None:
    """Called by the Upload Data page once its pipeline run finishes, to make
    the results visible to every other page via the getters above."""
    st.session_state[_SESSION_KEYS["combined_long"]] = combined_long
    st.session_state[_SESSION_KEYS["family_wide"]] = family_wide
    st.session_state[_SESSION_KEYS["phylum_wide"]] = phylum_wide
    st.session_state[_SESSION_KEYS["alpha_diversity"]] = alpha_diversity
    if core_microbiome is not None:
        st.session_state[_SESSION_KEYS["core_microbiome"]] = core_microbiome


def clear_uploaded_results() -> None:
    """Lets the Upload Data page offer a 'switch back to local data' reset."""
    for key in _SESSION_KEYS.values():
        st.session_state.pop(key, None)
