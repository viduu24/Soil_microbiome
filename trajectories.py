# src/trajectories.py
import sys

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats

# Resolve "processed/" relative to this script's own folder, not whatever
# directory the terminal happens to be in when streamlit is launched.
# wrangle.py used the same relative "processed" path and was run from
# src/, so it created src/processed/*.parquet — mirror that here.
SRC_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SRC_DIR / "processed"
sys.path.insert(0, str(SRC_DIR))

import data_source as ds

st.set_page_config(page_title="Taxonomic Abundance Trajectories", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_long_data() -> pd.DataFrame:
    # Reads uploaded/session data first (see pages/0_Upload_Data.py), falling
    # back to the local processed/*.parquet files -- NOT cached with
    # @st.cache_data, since that would ignore session_state changes (e.g.
    # switching from local data to an uploaded dataset mid-session).
    return ds.get_combined_long(PROCESSED_DIR)


def get_meta_cols(df: pd.DataFrame) -> list[str]:
    """Everything that isn't taxon / rel_abund / rank is metadata."""
    return [c for c in df.columns if c not in ("taxon", "rel_abund", "rank")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def top_n_taxa(df: pd.DataFrame, n: int, exclude_unclassified: bool) -> list[str]:
    """Rank taxa by mean relative abundance (within the already rank-filtered df)."""
    working = df if not exclude_unclassified else df[df["taxon"] != "Unclassified"]
    means = (
        working.groupby("taxon")["rel_abund"]
        .mean()
        .sort_values(ascending=False)
    )
    return means.head(n).index.tolist()


def top_n_taxa_per_facet(df: pd.DataFrame, n: int, exclude_unclassified: bool, facet_cols: list[str]) -> dict:
    """
    Computes top-N taxa (by mean relative abundance) INDEPENDENTLY within
    each combination of facet_cols values, rather than one global ranking
    applied uniformly to every facet panel. A taxon dominant in one Season
    but rare in another will only show up in the facet(s) where it's
    actually a top performer, instead of appearing (mostly flat/empty) in
    every panel just because it ranked highly overall.

    Returns {facet_key: [taxon, ...]}, where facet_key is a tuple of facet
    column values in the same order as `facet_cols`.
    """
    working = df if not exclude_unclassified else df[df["taxon"] != "Unclassified"]
    result = {}
    for facet_key, sub in working.groupby(facet_cols, dropna=False):
        if not isinstance(facet_key, tuple):
            facet_key = (facet_key,)
        means = sub.groupby("taxon")["rel_abund"].mean().sort_values(ascending=False)
        result[facet_key] = means.head(n).index.tolist()
    return result


def natural_stage_order(stages: list[str]) -> list[str]:
    """
    Best-effort ordering for decomposition stage labels. Falls back to
    alphabetical if labels don't match common decomposition-stage vocabulary,
    but the sidebar lets the user override this manually either way.
    """
    known_order = [
        "fresh", "bloat", "active decay", "active", "advanced decay",
        "advanced", "dry", "dry remains", "remains", "skeletal",
    ]
    def sort_key(s):
        s_low = str(s).strip().lower()
        for i, k in enumerate(known_order):
            if k in s_low:
                return (0, i, s_low)
        return (1, 0, s_low)
    return sorted(stages, key=sort_key)


def build_trajectory_df(
    df: pd.DataFrame,
    x_col: str,
    taxa: list[str],
    facet_row_col: str | None,
    facet_col_col: str | None,
    x_categories: list[str] | None = None,
    per_facet_taxa: dict | None = None,
) -> pd.DataFrame:
    """
    Collapses replicate samples into mean +/- SEM trajectories per
    (x value, taxon, facet_row, facet_col).

    When `x_categories` is given (i.e. x_col is the categorical "Stage"
    axis), the result is reindexed to include EVERY (x value x taxon x
    facet) combination, filling missing ones with 0 mean/SEM/n=0. Without
    this, a taxon with no samples at a given stage simply has no row for
    that stage, and the line plot draws a straight line connecting its
    nearest surrounding points -- skipping over the missing stage
    entirely and producing misleading diagonal "shortcut" lines that
    cross other taxa's trajectories. Filling with 0 makes "not detected
    here" show up as a real dip to zero, which is the accurate way to
    represent it. This reindex is skipped for the continuous ADH axis,
    where there's no fixed small set of x categories to fill gaps against.

    When `per_facet_taxa` is given (a dict from top_n_taxa_per_facet:
    {facet_key_tuple: [taxa,...]}), each facet gets built and reindexed
    using ONLY its own top taxa, rather than the single flat `taxa` list
    applied identically to every facet -- so a taxon dominant in one
    Season but rare in another only appears in the facet(s) where it's
    actually a top performer, instead of a flat near-zero line cluttering
    every panel. Requires facet_row_col and/or facet_col_col to be set.
    """
    group_cols = [x_col, "taxon"]
    facet_cols = []
    if facet_row_col:
        group_cols.append(facet_row_col)
        facet_cols.append(facet_row_col)
    if facet_col_col:
        group_cols.append(facet_col_col)
        facet_cols.append(facet_col_col)

    if per_facet_taxa is not None and facet_cols:
        pieces = []
        for facet_key, facet_taxa in per_facet_taxa.items():
            mask = pd.Series(True, index=df.index)
            for col, val in zip(facet_cols, facet_key):
                mask &= (df[col] == val)
            sub = df[mask & df["taxon"].isin(facet_taxa)].copy()
            if sub.empty:
                continue

            piece = (
                sub.groupby([x_col, "taxon"], dropna=False)["rel_abund"]
                .agg(mean="mean", sem="sem", n="count")
                .reset_index()
            )
            if x_categories:
                full_index = pd.MultiIndex.from_product(
                    [x_categories, facet_taxa], names=[x_col, "taxon"]
                )
                piece = piece.set_index([x_col, "taxon"]).reindex(full_index).reset_index()
                piece["n"] = piece["n"].fillna(0).astype(int)
                piece["mean"] = piece["mean"].fillna(0.0)
            for col, val in zip(facet_cols, facet_key):
                piece[col] = val
            pieces.append(piece)
        agg = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
            columns=group_cols + ["mean", "sem", "n"]
        )
    else:
        sub = df[df["taxon"].isin(taxa)].copy()
        agg = (
            sub.groupby(group_cols, dropna=False)["rel_abund"]
            .agg(mean="mean", sem="sem", n="count")
            .reset_index()
        )
        if x_categories:
            index_parts = [x_categories, taxa]
            if facet_row_col:
                index_parts.append(sorted(sub[facet_row_col].dropna().unique().tolist()))
            if facet_col_col:
                index_parts.append(sorted(sub[facet_col_col].dropna().unique().tolist()))

            full_index = pd.MultiIndex.from_product(index_parts, names=group_cols)
            agg = agg.set_index(group_cols).reindex(full_index).reset_index()
            agg["n"] = agg["n"].fillna(0).astype(int)
            agg["mean"] = agg["mean"].fillna(0.0)

    agg["sem"] = agg["sem"].fillna(0.0)
    agg["mean_pct"] = agg["mean"] * 100
    agg["sem_pct"] = agg["sem"] * 100
    agg["upper"] = agg["mean_pct"] + agg["sem_pct"]
    agg["lower"] = (agg["mean_pct"] - agg["sem_pct"]).clip(lower=0)
    return agg


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH FDR correction; NaN p-values stay NaN and are ignored in the count."""
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(pvals.shape, np.nan)
    ok = ~np.isnan(pvals)
    if ok.sum() == 0:
        return out
    p = pvals[ok]
    order = np.argsort(p)
    bh = p[order] * len(p) / (np.arange(len(p)) + 1)
    bh = np.clip(np.minimum.accumulate(bh[::-1])[::-1], 0, 1)
    res = np.empty(len(p))
    res[order] = bh
    out[ok] = res
    return out


def compute_trend_stats(
    df: pd.DataFrame,
    x_col: str,
    x_order: list[str] | None,
    taxa: list[str],
    facet_cols: list[str],
    per_facet_taxa: dict | None = None,
) -> pd.DataFrame:
    """
    Per-taxon (and per-facet) trend statistics on INDIVIDUAL SAMPLES, not on
    the plotted means, so replicate variation counts.

    - Continuous x (ADH / Days Since Death): linear regression of the taxon's
      relative abundance (%) on x -> slope, R^2, p.
    - Stage (categorical): stages are coded 0,1,2,... in the plotted order and
      regressed the same way (assumes evenly spaced steps), plus a one-way
      ANOVA eta^2 / p that makes no linearity assumption.
    - Spearman rho / p (monotonic trend) is reported for both.
    - q_value = Benjamini-Hochberg across every row in the table.

    Samples where a taxon has no row are counted as 0% (absent), not dropped --
    dropping them would bias every taxon toward the samples where it occurs.
    """
    is_stage = x_order is not None
    stage_idx = {s: i for i, s in enumerate(x_order)} if is_stage else None
    groups = df.groupby(facet_cols, dropna=False) if facet_cols else [((), df)]

    rows = []
    for key, sub in groups:
        if facet_cols and not isinstance(key, tuple):
            key = (key,)
        facet_taxa = per_facet_taxa.get(key, []) if per_facet_taxa is not None else taxa

        samp_x = sub.drop_duplicates("sample.id").set_index("sample.id")[x_col]
        x = samp_x.map(stage_idx) if is_stage else pd.to_numeric(samp_x, errors="coerce")
        mat = (
            sub.pivot_table(index="sample.id", columns="taxon", values="rel_abund",
                            aggfunc="sum", fill_value=0)
            .reindex(samp_x.index, fill_value=0) * 100
        )
        ok = x.notna().to_numpy()

        for taxon in facet_taxa:
            y = mat[taxon] if taxon in mat.columns else pd.Series(0.0, index=samp_x.index)
            xv, yv = x.to_numpy(float)[ok], y.to_numpy(float)[ok]
            row = dict(zip(facet_cols, key))
            row.update({"taxon": taxon, "n_samples": len(xv), "slope": np.nan, "R2": np.nan,
                        "p_value": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan})
            if is_stage:
                row.update({"eta2_anova": np.nan, "anova_p": np.nan})

            if len(xv) >= 3 and np.ptp(xv) > 0 and np.ptp(yv) > 0:
                lr = stats.linregress(xv, yv)
                rho, rho_p = stats.spearmanr(xv, yv)
                row.update({"slope": lr.slope, "R2": lr.rvalue ** 2, "p_value": lr.pvalue,
                            "spearman_rho": rho, "spearman_p": rho_p})
                if is_stage:
                    grand = yv.mean()
                    levels = [yv[xv == v] for v in np.unique(xv)]
                    ss_tot = ((yv - grand) ** 2).sum()
                    ss_bet = sum(len(g) * (g.mean() - grand) ** 2 for g in levels)
                    row["eta2_anova"] = ss_bet / ss_tot
                    if len(levels) >= 2 and len(yv) > len(levels):
                        row["anova_p"] = stats.f_oneway(*levels).pvalue
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = _benjamini_hochberg(out["p_value"].to_numpy())
    if is_stage:
        out["anova_q"] = _benjamini_hochberg(out["anova_p"].to_numpy())
    return out.sort_values(facet_cols + ["R2"], ascending=[True] * len(facet_cols) + [False],
                           na_position="last").reset_index(drop=True)


def plot_trajectories(
    agg: pd.DataFrame,
    x_col: str,
    x_order: list[str] | None,
    facet_row_col: str | None,
    facet_col_col: str | None,
    y_label: str,
    x_label: str,
    show_error_bars: bool = False,
) -> go.Figure:

    # Color palette for the taxa.
    # These colors are used in the interactive website plot.
    publication_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#393b79",
        "#637939",
        "#8c6d31",
        "#843c39",
        "#7b4173",
    ]

    fig = px.line(
        agg,
        x=x_col,
        y="mean_pct",
        color="taxon",
        facet_row=facet_row_col if facet_row_col else None,
        facet_col=facet_col_col if facet_col_col else None,
        error_y="sem_pct" if show_error_bars else None,
        markers=True,
        category_orders={
            x_col: x_order
        } if x_order else None,
        labels={
            "mean_pct": y_label,
            x_col: x_label,
            "taxon": "Taxon",
        },
        color_discrete_sequence=publication_colors,
    )

    fig.update_layout(
        height=550 if not (facet_row_col or facet_col_col) else 650,
        legend_title_text="Taxon",
        margin=dict(
            t=60,
            r=20,
            b=50,
            l=70,
        ),
    )

    # Keep the website's normal theme.
    # Only improve readability of text.
    fig.update_layout(
        font=dict(size=13),
    )

    fig.update_xaxes(
        title_font=dict(size=14),
        tickfont=dict(size=12),
    )

    fig.update_yaxes(
        title_font=dict(size=14),
        tickfont=dict(size=12),
    )

    # Slightly thicker colored lines
    fig.update_traces(
        line=dict(width=2.5),
        marker=dict(size=7),
    )

    # Clean facet titles
    fig.for_each_annotation(
        lambda a: a.update(
            text=a.text.split("=")[-1],
            font=dict(size=13),
        )
    )

    return fig
# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.title("Taxonomic Abundance Trajectories")
    st.caption(
        "Relative abundance trajectories across decomposition Stage or ADH, "
        "faceted by metadata groupings."
    )

    if not ds.data_available(PROCESSED_DIR):
        st.error(
            f"No data available -- either run wrangle.py locally (writes to "
            f"{PROCESSED_DIR}), or use the Upload Data page to use your own dataset."
        )
        return

    df_all = load_long_data()
    meta_cols = get_meta_cols(df_all)

    # --- Sidebar: rank ---
    st.sidebar.header("Data")
    rank = st.sidebar.radio("Taxonomic rank", options=["Family", "Phylum"], horizontal=True)
    df_rank = df_all[df_all["rank"] == rank].copy()

    exclude_unclassified = st.sidebar.checkbox("Exclude 'Unclassified'", value=True)
    if exclude_unclassified:
        df_rank = df_rank[df_rank["taxon"] != "Unclassified"]

    # --- Sidebar: x-axis toggle ---
    st.sidebar.header("X-axis")
    # "Days Since Death" only appears once wrangle.py's weather merge has
    # actually been run (see weather_merge.py / WEATHER_LOG_SOURCE) -- it
    # isn't present in combined_long.parquet until then, so it's only
    # offered here when the column is actually available.
    x_axis_options = ["Stage", "ADH"]
    if "DaysSinceDeath" in df_rank.columns:
        x_axis_options.append("Days Since Death")
    x_choice = st.sidebar.radio("X-axis variable", options=x_axis_options, horizontal=True)

    if x_choice == "Stage":
        if "Stage" not in df_rank.columns:
            st.error("Column 'Stage' not found in metadata.")
            return
        all_stages = df_rank["Stage"].dropna().unique().tolist()
        default_order = natural_stage_order(all_stages)
        x_order = st.sidebar.multiselect(
            "Stage order (drag not supported — pick in desired sequence)",
            options=all_stages,
            default=default_order,
        )
        if len(x_order) != len(all_stages):
            missing = [s for s in all_stages if s not in x_order]
            x_order = x_order + missing
        x_col = "Stage"
        x_label = "Decomposition Stage"
    elif x_choice == "ADH":
        if "ADH" not in df_rank.columns:
            st.error("Column 'ADH' not found in metadata.")
            return
        x_col = "ADH"
        x_label = "Accumulated Degree Hours (ADH)"
        x_order = None
    else:  # "Days Since Death"
        if "DaysSinceDeath" not in df_rank.columns:
            st.error(
                "Column 'DaysSinceDeath' not found. Run wrangle.py with "
                "WEATHER_LOG_SOURCE set to attach real calendar-day counts first."
            )
            return
        x_col = "DaysSinceDeath"
        x_label = "Days Since Death (calendar days, from daily weather log)"
        x_order = None

    # --- Sidebar: facet / grouping controls (moved above Taxa, since the
    # per-facet top-N option below needs to know which facets are active) ---
    st.sidebar.header("Grouping / Facets")
    facet_candidates = [c for c in ["Season", "Family"] if c in df_rank.columns]
    facet_options = ["None"] + facet_candidates
    # "Family" in the metadata means sampling position (Side vs. Under the
    # carcass), NOT the Family taxonomic rank -- the underlying column is
    # still literally named "Family" (a display label doesn't rename the
    # actual data), so give it a friendlier label here for the UI only,
    # matching the same fix already applied on the Diversity/ANCOM/PMI pages.
    FACET_LABELS = {"Family": "Position (Side/Under carcass)"}
    def _facet_label(c: str) -> str:
        return FACET_LABELS.get(c, c)

    facet_row_choice = st.sidebar.selectbox("Facet rows by", options=facet_options, format_func=_facet_label, index=0)
    facet_col_choice = st.sidebar.selectbox(
        "Facet columns by",
        options=[o for o in facet_options if o != facet_row_choice or o == "None"],
        format_func=_facet_label,
        index=0,
    )

    facet_row_col = None if facet_row_choice == "None" else facet_row_choice
    facet_col_col = None if facet_col_choice == "None" else facet_col_choice
    active_facet_cols = [c for c in [facet_row_col, facet_col_col] if c]

    # --- Sidebar: taxa selection ---
    st.sidebar.header("Taxa")
    max_n = max(1, df_rank["taxon"].nunique())
    top_n = st.sidebar.slider(
        "Top N most abundant taxa (by mean relative abundance)",
        min_value=1,
        max_value=min(30, max_n),
        value=min(8, max_n),
    )

    per_facet_mode = False
    per_facet_taxa = None
    if active_facet_cols:
        per_facet_mode = st.sidebar.checkbox(
            f"Compute top N independently per {' / '.join(_facet_label(c) for c in active_facet_cols)}",
            value=True,
            help="When on, each facet panel shows ITS OWN top-N taxa (e.g. Fall's top 8 "
                 "taxa can differ from Summer's top 8), instead of one global top-N "
                 "ranking applied identically to every panel. A taxon dominant in one "
                 "facet but rare in another will then only appear where it actually "
                 "matters, rather than cluttering every panel with a near-flat line.",
        )

    if per_facet_mode:
        per_facet_taxa = top_n_taxa_per_facet(df_rank, top_n, exclude_unclassified, active_facet_cols)
        selected_taxa = sorted({t for taxa_list in per_facet_taxa.values() for t in taxa_list})
        st.sidebar.caption(
            f"Showing each facet's own top {top_n} taxa ({len(selected_taxa)} distinct "
            f"taxa total across all facets). Manual taxa override is unavailable in this mode."
        )
    else:
        default_taxa = top_n_taxa(df_rank, top_n, exclude_unclassified)
        all_taxa_options = sorted(df_rank["taxon"].unique().tolist())
        selected_taxa = st.sidebar.multiselect(
            "Taxa to display (override — add or remove from top-N default)",
            options=all_taxa_options,
            default=default_taxa,
        )

    if not selected_taxa:
        st.warning("Select at least one taxon in the sidebar to plot.")
        return

    # --- Sidebar: display options ---
    st.sidebar.header("Display")
    show_error_bars = st.sidebar.checkbox(
        "Show error bars (SEM across replicates)", value=False
    )

    # --- Build aggregated trajectory data ---
    agg = build_trajectory_df(
        df_rank, x_col, selected_taxa, facet_row_col, facet_col_col,
        x_categories=x_order, per_facet_taxa=per_facet_taxa,
    )

    if agg.empty:
        st.warning("No data available for the current selection.")
        return

    # --- Plot ---
    fig = plot_trajectories(
        agg,
        x_col=x_col,
        x_order=x_order,
        facet_row_col=facet_row_col,
        facet_col_col=facet_col_col,
        y_label="Mean relative abundance (%)",
        x_label=x_label,
        show_error_bars=show_error_bars,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"trajectories_{rank}_{x_choice.replace(' ', '_')}",
                "height": 600,
                "width": 1000,
                "scale": 3,
            },
        },
    )

    # ------------------------------------------------------------------
    # Publication-quality figure downloads
    # ------------------------------------------------------------------

    st.subheader("Download figure")

    st.caption(
        "PNG is provided at high resolution. SVG and PDF are vector formats "
        "suitable for publication and further editing."
    )

    download_col1, download_col2, download_col3 = st.columns(3)

    # PNG
    try:
        png_bytes = fig.to_image(
            format="png",
            width=2100,
            height=1400,
            scale=1,
        )

        with download_col1:
            st.download_button(
                label="⬇ Download PNG",
                data=png_bytes,
                file_name=(
                    f"trajectories_{rank}_{x_choice.replace(' ', '_')}.png"
                ),
                mime="image/png",
                width="stretch",
            )

    except Exception as e:
        with download_col1:
            st.error("PNG export unavailable.")
            st.caption(str(e))


    # SVG
    try:
        svg_bytes = fig.to_image(
            format="svg",
            width=2100,
            height=1400,
        )

        with download_col2:
            st.download_button(
                label="⬇ Download SVG",
                data=svg_bytes,
                file_name=(
                    f"trajectories_{rank}_{x_choice.replace(' ', '_')}.svg"
                ),
                mime="image/svg+xml",
                width="stretch",
            )

    except Exception as e:
        with download_col2:
            st.error("SVG export unavailable.")
            st.caption(str(e))


    # PDF
    try:
        pdf_bytes = fig.to_image(
            format="pdf",
            width=2100,
            height=1400,
        )

        with download_col3:
            st.download_button(
                label="⬇ Download PDF",
                data=pdf_bytes,
                file_name=(
                    f"trajectories_{rank}_{x_choice.replace(' ', '_')}.pdf"
                ),
                mime="application/pdf",
                width="stretch",
            )

    except Exception as e:
        with download_col3:
            st.error("PDF export unavailable.")
            st.caption(str(e))


    # ------------------------------------------------------------------
    # Trend statistics (R^2 table)
    # ------------------------------------------------------------------

    st.subheader("Trend statistics (R\u00b2 table)")
    if x_choice == "Stage":
        st.caption(
            "Linear regression of each taxon's relative abundance (%) on stage order "
            "(stages coded 0, 1, 2, ... in the order chosen in the sidebar; assumes evenly "
            "spaced steps), plus a one-way ANOVA \u03b7\u00b2 that doesn't assume a straight-line "
            "trend. Computed on individual samples, not the plotted means; samples where "
            "a taxon is absent count as 0%. q = Benjamini-Hochberg across this table."
        )
    else:
        st.caption(
            f"Linear regression of each taxon's relative abundance (%) on {x_label}; "
            "slope is % abundance per unit of x. Computed on individual samples, not the "
            "plotted means; samples where a taxon is absent count as 0%. "
            "q = Benjamini-Hochberg across this table."
        )

    trend_df = compute_trend_stats(
        df_rank, x_col, x_order if x_choice == "Stage" else None,
        selected_taxa, active_facet_cols, per_facet_taxa,
    )
    if trend_df.empty:
        st.info("Not enough data to compute trend statistics for this selection.")
    else:
        n_min = int(trend_df["n_samples"].min())
        if n_min < 8:
            st.warning(
                f"Some rows have as few as {n_min} samples -- R\u00b2 from so few points is "
                f"unstable, treat those as exploratory."
            )
        st.dataframe(
            trend_df.rename(columns={"R2": "R\u00b2", "eta2_anova": "\u03b7\u00b2 (ANOVA)",
                                     "spearman_rho": "Spearman \u03c1"}),
            width="stretch", hide_index=True,
        )
        st.download_button(
            "Download trend statistics (CSV)",
            data=trend_df.to_csv(index=False).encode("utf-8"),
            file_name=f"trajectory_trend_stats_{rank}_{x_choice.replace(' ', '_')}.csv",
            mime="text/csv",
        )

    # --- Data table + download ---

    error_bar_note = "error bars = SEM across replicates" if show_error_bars else "error bars hidden (toggle in sidebar)"
    st.caption(
        f"n = {df_rank['sample.id'].nunique()} samples at {rank} rank | {error_bar_note}"
    )


if __name__ == "__main__":
    main()