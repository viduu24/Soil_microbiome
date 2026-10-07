# pages/5_Cooccurrence_Network.py
"""
Co-occurrence network page with a hierarchical Season -> Stage -> Position
drill-down: pick a Season tab, then optionally narrow further to a single
Stage, then optionally narrow further to a single sampling Position
(Side/Under the carcass -- the "Family" metadata column, per the same
Position-not-taxonomy convention used on the Diversity/PMI pages).

Each narrower slice is a SMALLER set of samples, and correlation-based
networks need enough samples to say anything meaningful -- a 4-sample
Season x Stage x Position slice will happily produce "significant" edges
that are really just noise. See the sample-size caption/warning below;
it's advisory, not a hard block, since you know your own data's limits
better than a fixed cutoff can.
"""

from pathlib import Path
import sys

import streamlit as st
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
PROCESSED_DIR = SRC_DIR / "processed"

from data_source import data_available, get_wide
import cooccurence as co
from export_utils import add_figure_download_buttons

# Below this many samples, a correlation-based network is unreliable enough
# to warrant a visible caution -- still computed if the researcher wants to
# look, just flagged. Purely a UI cue, not a statistical cutoff.
MIN_RECOMMENDED_SAMPLES = 15

st.set_page_config(page_title="Co-occurrence Network", layout="wide")
st.title("Taxon Co-occurrence Network")

if not data_available(PROCESSED_DIR):
    st.warning("No processed data found. Run `wrangle.py` first, or use the Upload Data page.")
    st.stop()

st.markdown(
    "Builds a network of taxa whose abundances rise and fall together across "
    "samples. See the module docstring in `cooccurrence.py` for why this "
    "defaults to a CLR-transformed correlation rather than a plain one."
)

col_a, col_b = st.columns(2)
with col_a:
    rank = st.selectbox("Taxonomic rank", ["Family", "Phylum"])
with col_b:
    method = st.selectbox(
        "Correlation method",
        ["clr_pearson", "spearman"],
        format_func=lambda m: "CLR + Pearson (recommended)" if m == "clr_pearson" else "Spearman (plain)",
    )

st.sidebar.header("Network thresholds")
min_prevalence = st.sidebar.slider("Minimum taxon prevalence", 0.0, 1.0, 0.3, 0.05,
                                    help="Drop taxa present in fewer than this fraction of samples.")
min_abs_corr = st.sidebar.slider("Minimum |correlation|", 0.0, 1.0, 0.6, 0.05)
alpha = st.sidebar.slider("Significance threshold (BH-adjusted q)", 0.001, 0.10, 0.05, 0.001)

wide = get_wide(PROCESSED_DIR, rank)
relabund_cols = [c for c in wide.columns if c.endswith("_relabund")]
taxon_names = [c.replace("_relabund", "") for c in relabund_cols]

# wide already carries its metadata columns alongside counts/relabund
# (see wrangle.py's compute_relative_abundance) -- "Family" here is sampling
# Position (Side/Under), not the Family taxonomic rank.
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


def _render_network(sub_abundance: pd.DataFrame, label: str, key_prefix: str):
    n_samples = len(sub_abundance)
    if n_samples < MIN_RECOMMENDED_SAMPLES:
        st.warning(
            f"Only {n_samples} samples in this slice -- below the {MIN_RECOMMENDED_SAMPLES} "
            f"recommended for a stable correlation network. Edges below are still computed "
            f"if you want to look, but treat them as exploratory, not publication-ready."
        )

    try:
        edges = co.compute_cooccurrence_edges(
            sub_abundance, method=method, min_prevalence=min_prevalence,
            alpha=alpha, min_abs_corr=min_abs_corr,
        )
    except ValueError as e:
        st.error(f"{label}: {e}")
        return

    if edges.empty:
        st.info(f"{label}: no taxon pairs cleared both thresholds ({n_samples} samples). "
                f"Try lowering 'Minimum |correlation|' or raising the significance threshold in the sidebar.")
        return

    st.caption(f"{n_samples} samples \u2192 {len(edges)} significant edges among "
               f"{len(set(edges['taxon_1']) | set(edges['taxon_2']))} taxa "
               f"({(edges['sign'] == 'positive').sum()} positive, "
               f"{(edges['sign'] == 'negative').sum()} negative)")

    G = co.build_graph(edges, wide=sub_abundance)
    fig = co.plot_cooccurrence_network(G, title=f"{rank}-level co-occurrence -- {label}")

    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        key=f"cooc_{key_prefix}_chart",
    )
    add_figure_download_buttons(
        fig, filename=f"cooccurrence_{rank.lower()}_{key_prefix}", key_prefix=f"cooc_{key_prefix}",
        width=1800, height=1200, scale=3,  # bigger canvas = more room for labels
    )

    with st.expander(f"Edge table -- {label}"):
        st.dataframe(edges, use_container_width=True)
        st.download_button(
            "Download edge table (CSV)",
            data=edges.to_csv(index=False).encode("utf-8"),
            file_name=f"cooccurrence_edges_{rank.lower()}_{key_prefix}.csv",
            mime="text/csv",
            key=f"cooc_{key_prefix}_csv",
        )


ALL_LABEL = "All"
ALL_SEASONS_LABEL = "All Seasons"

seasons = sorted(meta["Season"].dropna().unique().tolist())
tab_labels = [ALL_SEASONS_LABEL] + seasons
season_tabs = st.tabs(tab_labels)

for season_tab, season in zip(season_tabs, tab_labels):
    with season_tab:
        # "All Seasons" pools every sample regardless of Season -- everything
        # below (Stage/Position drill-down, network) works identically on
        # top of that pooled set, it's just not restricted to one Season first.
        if season == ALL_SEASONS_LABEL:
            season_mask = pd.Series(True, index=meta.index)
        else:
            season_mask = meta["Season"] == season
        season_ids = meta.index[season_mask]

        # --- Stage drill-down (optional) ---
        if has_stage:
            stages_in_season = sorted(meta.loc[season_ids, "Stage"].dropna().unique().tolist())
            stage_choice = st.selectbox(
                "Stage", [ALL_LABEL] + stages_in_season, key=f"stage_{season}",
            )
        else:
            stage_choice = ALL_LABEL

        stage_ids = season_ids if stage_choice == ALL_LABEL else meta.index[
            season_mask & (meta["Stage"] == stage_choice)
        ]

        # --- Position drill-down (Side/Under), optional ---
        if has_position:
            positions_here = sorted(meta.loc[stage_ids, POSITION_COL].dropna().unique().tolist())
            position_choice = st.selectbox(
                "Position (Side/Under)", [ALL_LABEL] + positions_here, key=f"position_{season}_{stage_choice}",
            )
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
        _render_network(sub_abundance, label=label, key_prefix=key_prefix)