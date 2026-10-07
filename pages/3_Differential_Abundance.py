# src/pages/2_Differential_Abundance.py
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
PROCESSED_DIR = SRC_DIR / "processed"

import streamlit as st
import pandas as pd
import plotly.express as px

import diversity as div
import data_source as ds
from export_utils import add_figure_download_buttons

st.set_page_config(page_title="Differential Abundance (ANCOM)", layout="wide")
st.title("Differential Abundance -- ANCOM")
st.caption(
    "Analysis of Composition of Microbiomes (Mandal et al. 2015): tests log-ratios "
    "between every pair of taxa across your chosen grouping variable, which is the "
    "compositionally-correct way to find taxa that differ between groups (a plain "
    "t-test on relative abundance is not valid here, since the values are constrained "
    "to sum to 1 per sample)."
)


def load_wide(rank: str) -> pd.DataFrame:
    return ds.get_wide(PROCESSED_DIR, rank)


if not ds.data_available(PROCESSED_DIR):
    st.error("No data available -- run wrangle.py locally, or use the Upload Data page.")
    st.stop()

st.sidebar.header("Data")
rank = st.sidebar.radio("Taxonomic rank", options=["Family", "Phylum"], horizontal=True)
wide_df = load_wide(rank)

relabund_cols = [c for c in wide_df.columns if c.endswith("_relabund")]
count_cols = [c.replace("_relabund", "") for c in relabund_cols if c.replace("_relabund", "") in wide_df.columns]
meta_cols = [c for c in wide_df.columns if c not in relabund_cols and c not in count_cols]
group_candidates = [c for c in ["Season", "Stage", "Family"] if c in meta_cols]
# "Family" here means sampling position (Side vs. Under the carcass), not the
# Family taxonomic rank selected above -- label it clearly in the UI.
GROUP_LABELS = {"Family": "Position (Side/Under carcass)"}
def _group_label(c: str) -> str:
    return GROUP_LABELS.get(c, c)

if not count_cols:
    st.error("No raw count columns found alongside the *_relabund columns in this export.")
    st.stop()

group_col = st.sidebar.selectbox("Group by", options=group_candidates, format_func=_group_label)
alpha = st.sidebar.slider("Significance level (alpha)", 0.01, 0.20, 0.05, step=0.01)
w_threshold = st.sidebar.slider(
    "W ratio threshold for 'detected'", 0.5, 0.95, 0.7, step=0.05,
    help="Fraction of pairwise log-ratio tests that must be significant for a "
         "taxon to be declared differentially abundant (0.7 matches the papers' default).",
)

groups_present = wide_df[group_col].dropna().unique()
if len(groups_present) < 2:
    st.warning(f"Need at least 2 groups in '{group_col}' to run ANCOM.")
    st.stop()

if st.button("Run ANCOM", type="primary"):
    counts = wide_df.set_index("sample.id")[count_cols]
    grouping = wide_df.set_index("sample.id")[group_col]
    with st.spinner(f"Running ANCOM across {len(count_cols)} taxa x {len(counts)} samples..."):
        result = div.ancom(counts, grouping, alpha=alpha, w_threshold=w_threshold)
    st.session_state["ancom_result"] = result
    st.session_state["ancom_rank"] = rank
    st.session_state["ancom_group_col"] = group_col

if "ancom_result" in st.session_state:
    result = st.session_state["ancom_result"]
    n_detected = int(result["detected"].sum())
    st.subheader(
        f"{n_detected} differentially abundant taxa found "
        f"({st.session_state['ancom_rank']} rank, grouped by {_group_label(st.session_state['ancom_group_col'])})"
    )

    fig = px.bar(
        result.reset_index().sort_values("W_ratio", ascending=True).tail(30),
        x="W_ratio",
        y="taxon",
        color="detected",
        orientation="h",
        labels={
            "W_ratio": "W statistic (fraction of significant pairwise tests)",
            "taxon": "Taxon",
            "detected": "Differentially abundant",
        },
        height=max(500, 25 * min(30, len(result))),
        color_discrete_map={
            True: "#2E86AB",
            False: "#B8C6D1",
        },
    )

    # Website appearance
    # Keep the normal Streamlit/Plotly background.
    fig.update_layout(
        margin=dict(l=200, r=40, t=40, b=60),
        font=dict(size=13),
    )

    # Display ONLY ONE graph on the website
    st.plotly_chart(
        fig,
        width="stretch",
        key="ancom_bar_chart",
    )

    # Download button only -- this does NOT create another graph
    add_figure_download_buttons(
        fig,
        filename=f"ancom_{st.session_state['ancom_rank']}_by_{st.session_state['ancom_group_col']}",
        key_prefix="ancom_bar",
    )
    st.dataframe(result.reset_index(), width="stretch")
    csv = result.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download ANCOM results (CSV)", data=csv, file_name=f"ancom_{rank}_{group_col}.csv")
else:
    st.info("Choose a grouping variable and click **Run ANCOM** -- this runs a Kruskal-Wallis "
            "test on every taxon pair so it can take a few seconds for larger taxa sets.")