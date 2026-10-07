# pages/6_Taxa_Heatmap.py
"""
Sample x taxon heatmaps with the same Season -> Stage -> Position drill-down
as the Co-occurrence page. The "All Seasons" tab is the overall view; each
Season tab can be narrowed to a Stage and then to a sampling Position
("Family" metadata column = Side/Under the carcass, not the taxonomic rank).
"""

from pathlib import Path
import sys

import streamlit as st
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
PROCESSED_DIR = SRC_DIR / "processed"

from data_source import data_available, get_wide
import taxa_heatmap as hm
from export_utils import add_figure_download_buttons

st.set_page_config(page_title="Taxa Heatmap", layout="wide")
st.title("Taxa x Sample Heatmap")

if not data_available(PROCESSED_DIR):
    st.warning("No processed data found. Run `wrangle.py` first, or use the Upload Data page.")
    st.stop()

col_a, col_b = st.columns(2)
with col_a:
    rank = st.selectbox("Taxonomic rank", ["Family", "Phylum"])
with col_b:
    transform = st.selectbox("Cell value", list(hm.TRANSFORMS), format_func=hm.TRANSFORMS.get,
                             help="log10 is the best default: a few dominant taxa otherwise wash out the rest.")

st.sidebar.header("Heatmap options")
top_n = st.sidebar.slider("Number of taxa (top by mean abundance)", 5, 100, 30, 5)
cluster_samples = st.sidebar.checkbox("Cluster samples (instead of Season/Stage/Position order)", False)
cluster_taxa = st.sidebar.checkbox("Cluster taxa (instead of abundance order)", False)

wide = get_wide(PROCESSED_DIR, rank)
relabund_cols = [c for c in wide.columns if c.endswith("_relabund")]
taxon_names = [c.replace("_relabund", "") for c in relabund_cols]

POSITION_COL = "Family"
has_season = "Season" in wide.columns
has_stage = "Stage" in wide.columns
has_position = POSITION_COL in wide.columns

if not has_season:
    st.error("No 'Season' column found in the processed data -- can't build the Season drill-down.")
    st.stop()

if relabund_cols:
    abundance_table = wide.set_index("sample.id")[relabund_cols].rename(columns=lambda c: c.replace("_relabund", ""))
else:
    abundance_table = wide.set_index("sample.id")[taxon_names]

meta = wide.set_index("sample.id")[[c for c in ["Season", "Stage", POSITION_COL] if c in wide.columns]]


def _render_heatmap(sub_abundance: pd.DataFrame, label: str, key_prefix: str):
    n_samples = len(sub_abundance)
    if n_samples < 2:
        st.info(f"{label}: only {n_samples} sample -- a heatmap needs at least 2.")
        return

    try:
        matrix = hm.prepare_matrix(
            sub_abundance, meta=meta, top_n=top_n, transform=transform,
            cluster_samples=cluster_samples, cluster_taxa=cluster_taxa,
        )
    except ValueError as e:
        st.error(f"{label}: {e}")
        return

    st.caption(f"{matrix.shape[1]} samples x {matrix.shape[0]} taxa")

    fig = hm.plot_taxa_heatmap(matrix, meta=meta, transform=transform,
                               title=f"{rank}-level heatmap -- {label}")
    st.plotly_chart(fig, use_container_width=True, theme=None, key=f"hm_{key_prefix}_chart")

    # Export canvas grows with the matrix so every taxon/sample label stays legible.
    w = int(min(3000, max(1400, 30 * matrix.shape[1] + 500)))
    h = int(min(2400, max(900, 32 * matrix.shape[0] + 300)))
    add_figure_download_buttons(
        fig, filename=f"heatmap_{rank.lower()}_{transform}_{key_prefix}",
        key_prefix=f"hm_{key_prefix}", width=w, height=h,
        scale=3 if w * h <= 3_500_000 else 2,
    )

    with st.expander(f"Matrix -- {label}"):
        st.dataframe(matrix, use_container_width=True)
        st.download_button(
            "Download matrix (CSV)",
            data=matrix.to_csv().encode("utf-8"),
            file_name=f"heatmap_matrix_{rank.lower()}_{transform}_{key_prefix}.csv",
            mime="text/csv",
            key=f"hm_{key_prefix}_csv",
        )


ALL_LABEL = "All"
ALL_SEASONS_LABEL = "All Seasons"

seasons = sorted(meta["Season"].dropna().unique().tolist())
tab_labels = [ALL_SEASONS_LABEL] + seasons
season_tabs = st.tabs(tab_labels)

for season_tab, season in zip(season_tabs, tab_labels):
    with season_tab:
        if season == ALL_SEASONS_LABEL:
            season_mask = pd.Series(True, index=meta.index)
        else:
            season_mask = meta["Season"] == season
        season_ids = meta.index[season_mask]

        if has_stage:
            stages_in_season = sorted(meta.loc[season_ids, "Stage"].dropna().unique().tolist())
            stage_choice = st.selectbox("Stage", [ALL_LABEL] + stages_in_season, key=f"hm_stage_{season}")
        else:
            stage_choice = ALL_LABEL

        stage_ids = season_ids if stage_choice == ALL_LABEL else meta.index[
            season_mask & (meta["Stage"] == stage_choice)
        ]

        if has_position:
            positions_here = sorted(meta.loc[stage_ids, POSITION_COL].dropna().unique().tolist())
            position_choice = st.selectbox("Position (Side/Under)", [ALL_LABEL] + positions_here,
                                           key=f"hm_position_{season}_{stage_choice}")
        else:
            position_choice = ALL_LABEL

        final_ids = stage_ids if position_choice == ALL_LABEL else [
            sid for sid in stage_ids if meta.loc[sid, POSITION_COL] == position_choice
        ]

        label_parts = ["All Seasons"] if season == ALL_SEASONS_LABEL else [f"Season={season}"]
        if stage_choice != ALL_LABEL:
            label_parts.append(f"Stage={stage_choice}")
        if position_choice != ALL_LABEL:
            label_parts.append(f"Position={position_choice}")
        label = ", ".join(label_parts)
        key_prefix = "_".join(part.replace("=", "-").replace(" ", "_") for part in label_parts)

        sub_abundance = abundance_table.loc[abundance_table.index.intersection(final_ids)]
        _render_heatmap(sub_abundance, label=label, key_prefix=key_prefix)