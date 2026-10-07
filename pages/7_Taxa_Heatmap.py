# pages/6_Taxa_Heatmap.py
"""
Sample x taxon heatmaps with the same Season -> Stage -> Position drill-down
as the Co-occurrence page. Two views:

  * Explore  -- the original layout: an "All Seasons" tab plus one tab per
                Season, each narrowable to a Stage and then a Position.
  * Compare  -- two heatmaps side by side. Each side gets its own
                Season / Stage / Position selection. Both sides use the same
                taxa, the same taxon order and the same color scale, so they
                can be compared directly.

"Position" is the "Family" metadata column = Side/Under the carcass, not the
taxonomic rank.
"""

from pathlib import Path
import sys

import numpy as np
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

ALL_LABEL = "All"
ALL_SEASONS_LABEL = "All Seasons"
seasons = sorted(meta["Season"].dropna().unique().tolist())
tab_labels = [ALL_SEASONS_LABEL] + seasons


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _export_size(matrix: pd.DataFrame):
    """Export canvas grows with the matrix so every taxon/sample label stays legible."""
    w = int(min(3000, max(1400, 30 * matrix.shape[1] + 500)))
    h = int(min(2400, max(900, 32 * matrix.shape[0] + 300)))
    return w, h


def _select_subset(season: str, ns: str):
    """Stage -> Position drill-down inside one Season.

    `ns` namespaces the widget keys so the same selectors can appear several
    times on the page (Explore tabs, Compare side A, Compare side B).
    Returns (sample_ids, label, key_prefix).
    """
    if season == ALL_SEASONS_LABEL:
        season_ids = meta.index
        label_parts = ["All Seasons"]
    else:
        season_ids = meta.index[(meta["Season"] == season).to_numpy()]
        label_parts = [f"Season={season}"]

    stage_choice = ALL_LABEL
    stage_ids = season_ids
    if has_stage:
        stages = sorted(meta.loc[season_ids, "Stage"].dropna().unique().tolist())
        stage_choice = st.selectbox("Stage", [ALL_LABEL] + stages, key=f"{ns}_stage_{season}")
        if stage_choice != ALL_LABEL:
            stage_ids = season_ids[(meta.loc[season_ids, "Stage"] == stage_choice).to_numpy()]
            label_parts.append(f"Stage={stage_choice}")

    final_ids = stage_ids
    if has_position:
        positions = sorted(meta.loc[stage_ids, POSITION_COL].dropna().unique().tolist())
        position_choice = st.selectbox("Position (Side/Under)", [ALL_LABEL] + positions,
                                       key=f"{ns}_position_{season}_{stage_choice}")
        if position_choice != ALL_LABEL:
            final_ids = stage_ids[(meta.loc[stage_ids, POSITION_COL] == position_choice).to_numpy()]
            label_parts.append(f"Position={position_choice}")

    label = ", ".join(label_parts)
    key_prefix = "_".join(part.replace("=", "-").replace(" ", "_") for part in label_parts)
    return list(final_ids), label, key_prefix


def _matrix_expander(matrix: pd.DataFrame, label: str, key_prefix: str):
    with st.expander(f"Matrix -- {label}"):
        st.dataframe(matrix, use_container_width=True)
        st.download_button(
            "Download matrix (CSV)",
            data=matrix.to_csv().encode("utf-8"),
            file_name=f"heatmap_matrix_{rank.lower()}_{transform}_{key_prefix}.csv",
            mime="text/csv",
            key=f"hm_{key_prefix}_csv",
        )


# ---------------------------------------------------------------------------
# Explore view (original behaviour)
# ---------------------------------------------------------------------------
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

    w, h = _export_size(matrix)
    add_figure_download_buttons(
        fig, filename=f"heatmap_{rank.lower()}_{transform}_{key_prefix}",
        key_prefix=f"hm_{key_prefix}", width=w, height=h,
        scale=3 if w * h <= 3_500_000 else 2,
    )
    _matrix_expander(matrix, label, key_prefix)


def explore_view():
    season_tabs = st.tabs(tab_labels)
    for season_tab, season in zip(season_tabs, tab_labels):
        with season_tab:
            ids, label, key_prefix = _select_subset(season, "hm")
            sub_abundance = abundance_table.loc[abundance_table.index.intersection(ids)]
            _render_heatmap(sub_abundance, label=label, key_prefix=key_prefix)


# ---------------------------------------------------------------------------
# Compare view (two groups side by side)
# ---------------------------------------------------------------------------
def _apply_shared_scale(fig, matrix: pd.DataFrame, zmin: float, zmax: float):
    """Force the same color range on the main heatmap trace of `fig`.

    Only traces whose z-array has the matrix's shape are touched, so any
    Season/Stage/Position annotation strips drawn by the plot function keep
    their own colors.
    """
    for tr in fig.data:
        if getattr(tr, "type", None) != "heatmap":
            continue
        if np.shape(tr.z) not in (matrix.shape, matrix.T.shape):
            continue
        if getattr(tr, "coloraxis", None):
            fig.update_layout({tr.coloraxis: dict(cmin=zmin, cmax=zmax)})
        else:
            tr.update(zmin=zmin, zmax=zmax)


def compare_view():
    st.caption(
        "Pick a subset for each side. Both heatmaps use the same taxa, the same "
        "taxon order and the same color scale, so differences you see are real "
        "differences between the groups."
    )
    if cluster_taxa:
        st.caption("Taxon clustering is switched off in this view so the rows line up across both sides.")

    # --- selectors, one column per side ---
    selections = []
    sel_cols = st.columns(2)
    for col, side in zip(sel_cols, ("A", "B")):
        with col:
            st.subheader(f"Side {side}")
            season = st.selectbox("Season", tab_labels, key=f"cmp{side}_season")
            ids, label, kp = _select_subset(season, f"cmp{side}")
            selections.append((ids, label, f"cmp{side}_{kp}"))

    subs = [abundance_table.loc[abundance_table.index.intersection(ids)] for ids, _, _ in selections]
    for (_, label, _), sub in zip(selections, subs):
        if len(sub) < 2:
            st.info(f"{label}: only {len(sub)} sample -- a heatmap needs at least 2.")
            return

    # --- shared taxa: top N by the average of the two group means ---
    group_means = (subs[0].mean() + subs[1].mean()) / 2
    shared = group_means.nlargest(top_n).index.tolist()

    matrices = []
    for sub, (_, label, _) in zip(subs, selections):
        try:
            m = hm.prepare_matrix(
                sub[shared], meta=meta, top_n=len(shared), transform=transform,
                cluster_samples=cluster_samples, cluster_taxa=False,
            )
        except ValueError as e:
            st.error(f"{label}: {e}")
            return
        matrices.append(m.loc[[t for t in shared if t in m.index]])  # same row order on both sides

    # --- shared color range ---
    values = np.concatenate([m.to_numpy(dtype=float).ravel() for m in matrices])
    zmin, zmax = (float(np.nanmin(values)), float(np.nanmax(values))) if np.isfinite(values).any() else (None, None)

    # --- plots, one column per side ---
    plot_cols = st.columns(2)
    for col, m, (_, label, kp) in zip(plot_cols, matrices, selections):
        with col:
            st.caption(f"{label} -- {m.shape[1]} samples x {m.shape[0]} taxa")
            fig = hm.plot_taxa_heatmap(m, meta=meta, transform=transform, title=label)
            if zmin is not None:
                _apply_shared_scale(fig, m, zmin, zmax)
            st.plotly_chart(fig, use_container_width=True, theme=None, key=f"hm_{kp}_chart")

            w, h = _export_size(m)
            add_figure_download_buttons(
                fig, filename=f"heatmap_{rank.lower()}_{transform}_{kp}",
                key_prefix=f"hm_{kp}", width=w, height=h,
                scale=3 if w * h <= 3_500_000 else 2,
            )
            _matrix_expander(m, label, kp)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
view_mode = st.radio("View", ["Explore", "Compare two groups side by side"], horizontal=True)

if view_mode == "Explore":
    explore_view()
else:
    compare_view()