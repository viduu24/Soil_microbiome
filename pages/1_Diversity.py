# src/pages/1_Diversity.py
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
PROCESSED_DIR = SRC_DIR / "processed"

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import diversity as div
import data_source as ds
from export_utils import add_figure_download_buttons


st.set_page_config(page_title="Alpha & Beta Diversity", layout="wide")
st.title("Alpha & Beta Diversity")
st.caption(
    "Richness/evenness within samples (alpha) and community dissimilarity "
    "between samples (beta), computed on the rarefied counts wrangle.py exports."
)


def load_alpha():
    return ds.get_alpha_diversity(PROCESSED_DIR)


def load_wide(rank: str) -> pd.DataFrame:
    return ds.get_wide(PROCESSED_DIR, rank)


if not ds.data_available(PROCESSED_DIR):
    st.error("No data available -- run wrangle.py locally, or use the Upload Data page.")
    st.stop()

alpha_df = load_alpha()
meta_grouping_cols = [c for c in ["Season", "Stage", "Family", "Carcass"] if c in alpha_df.columns]
# Continuous (numeric) variables suitable for a scatter-style "diversity over
# time" view, as opposed to the categorical box-plot grouping above.
# DaysSinceDeath only exists once wrangle.py's weather merge has been run.
numeric_candidates = [c for c in ["ADH", "DaysSinceDeath"] if c in alpha_df.columns]
# "Family" in the metadata means sampling position (Side vs. Under the carcass),
# NOT the Family taxonomic rank -- give it a clearer label in the UI only.
GROUP_LABELS = {"Family": "Position (Side/Under carcass)", "DaysSinceDeath": "Days Since Death"}
def _group_label(c: str) -> str:
    return GROUP_LABELS.get(c, c)

st.sidebar.header("Data")
rank = st.sidebar.radio("Taxonomic rank", options=["Family", "Phylum"], horizontal=True)
alpha_rank = alpha_df[alpha_df["rank"] == rank]

# --- Alpha diversity ---
st.header("Alpha diversity")
metric = st.sidebar.radio(
    "Alpha metric", options=["observed_richness", "chao1", "shannon"], horizontal=True
)

alpha_view = st.sidebar.radio(
    "Alpha diversity view",
    options=["By category (box plot)", "By continuous variable (scatter)"] if numeric_candidates else ["By category (box plot)"],
    help="'By continuous variable' plots diversity against a numeric time proxy "
         "(ADH or Days Since Death) instead of grouping samples into categories.",
)

if alpha_view == "By category (box plot)":
    group_col = st.sidebar.selectbox("Group alpha diversity by", options=meta_grouping_cols, format_func=_group_label)
    fig_alpha = px.box(
        alpha_rank, x=group_col, y=metric, color=group_col, points="all",
        labels={metric: metric.replace("_", " ").title()},
    )
    fig_alpha.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig_alpha, use_container_width=True)
    add_figure_download_buttons(fig_alpha, filename=f"alpha_diversity_{metric}_by_{group_col}", key_prefix="alpha_box")
    table_cols = ["sample.id", group_col, "observed_richness", "chao1", "shannon"]
else:
    x_col = st.sidebar.selectbox("X-axis (continuous)", options=numeric_candidates, format_func=_group_label)
    color_choice = st.sidebar.selectbox(
        "Color by (optional category)", options=["None"] + meta_grouping_cols, format_func=_group_label,
    )
    color_arg = None if color_choice == "None" else color_choice
    fig_alpha = px.scatter(
        alpha_rank, x=x_col, y=metric, color=color_arg,
        labels={metric: metric.replace("_", " ").title(), x_col: _group_label(x_col)},
    )
    # Simple linear trend line (scipy, already a dependency -- avoids needing
    # statsmodels just for Plotly Express's built-in trendline option).
    trend_data = alpha_rank[[x_col, metric]].dropna()
    if len(trend_data) >= 2:
        from scipy.stats import linregress
        slope, intercept, r_value, _, _ = linregress(trend_data[x_col], trend_data[metric])
        x_range = [trend_data[x_col].min(), trend_data[x_col].max()]
        fig_alpha.add_trace(go.Scatter(
            x=x_range, y=[slope * xv + intercept for xv in x_range],
            mode="lines", name=f"Linear trend (R²={r_value**2:.2f})",
            line=dict(color="firebrick", dash="dash"),
        ))
    fig_alpha.update_layout(height=450)
    st.plotly_chart(fig_alpha, width="stretch")
    add_figure_download_buttons(fig_alpha, filename=f"alpha_diversity_{metric}_vs_{x_col}", key_prefix="alpha_scatter")
    table_cols = ["sample.id", x_col, "observed_richness", "chao1", "shannon"]

with st.expander("View alpha diversity table"):
    st.dataframe(alpha_rank[table_cols], width="stretch")

st.divider()

# --- Beta diversity ---
st.header("Beta diversity (PCoA)")
wide_df = load_wide(rank)
# Raw count columns are the ones with a matching "<name>_relabund" sibling.
relabund_cols = [c for c in wide_df.columns if c.endswith("_relabund")]
count_cols = [c.replace("_relabund", "") for c in relabund_cols]
count_cols = [c for c in count_cols if c in wide_df.columns]
counts_wide = wide_df.set_index("sample.id")[count_cols] if count_cols else None

MAX_SHAPE_CATEGORIES = 6  # shape only stays readable with a handful of distinct categories
DISTINCT_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x"]

beta_metric = st.sidebar.radio("Beta metric", options=["braycurtis", "jaccard"], horizontal=True)
# "Color PCoA by" now offers both categorical columns (discrete legend) and
# continuous numeric ones like ADH/DaysSinceDeath (colorbar gradient) --
# which branch runs is decided below based on the chosen column's dtype.
color_options = meta_grouping_cols + [c for c in numeric_candidates if c in wide_df.columns]
beta_group_col = st.sidebar.selectbox("Color PCoA by", options=color_options, format_func=_group_label, key="beta_group")
beta_color_is_numeric = pd.api.types.is_numeric_dtype(wide_df[beta_group_col])

# Second grouping dimension, shown via marker SHAPE rather than color, so two
# variables can be inspected on the same plot at once (e.g. color = Stage,
# shape = Position) -- defaults to Position (Side/Under) since that's the
# comparison most likely to matter alongside whatever is chosen for color.
# Restricted to low-cardinality columns (<= MAX_SHAPE_CATEGORIES unique values):
# beyond a handful of categories, distinct marker shapes stop being visually
# distinguishable and Plotly starts silently reusing symbols, which is
# actively misleading rather than just cluttered -- so high-cardinality
# columns like Carcass are excluded from this dropdown (still fine for color,
# which scales to more categories more gracefully).
shape_candidates = [c for c in meta_grouping_cols if wide_df[c].nunique(dropna=True) <= MAX_SHAPE_CATEGORIES]
shape_options = ["None"] + shape_candidates
default_shape_index = shape_options.index("Family") if "Family" in shape_candidates else 0
shape_col_choice = st.sidebar.selectbox(
    "Shape PCoA by", options=shape_options, index=default_shape_index,
    format_func=_group_label,
    help=f"Adds a second grouping variable to the same plot, shown as marker shape "
         f"instead of color, so you can see two dimensions (e.g. Stage and Position) at "
         f"once. Limited to variables with {MAX_SHAPE_CATEGORIES} or fewer categories, "
         f"since more than that becomes hard to tell apart by shape alone.",
)
shape_col = None if shape_col_choice == "None" else shape_col_choice

if counts_wide is not None and len(counts_wide) > 2:
    dist = div.beta_diversity(counts_wide, metric=beta_metric)
    coords, var_exp = div.pcoa(dist)
    meta_lookup_cols = sorted(set(meta_grouping_cols + numeric_candidates) & set(wide_df.columns))
    meta_lookup = wide_df.set_index("sample.id")[meta_lookup_cols]
    coords = coords.join(meta_lookup)

    if shape_col:
        shape_categories = sorted(coords[shape_col].dropna().unique().tolist())
        symbol_map = {cat: DISTINCT_SYMBOLS[i % len(DISTINCT_SYMBOLS)] for i, cat in enumerate(shape_categories)}
    else:
        shape_categories, symbol_map = [], {}

    fig_pcoa = go.Figure()

    if beta_color_is_numeric:
        # Continuous coloring: a single shared color scale (colorbar) across
        # every point, rather than discrete per-category colors. If a shape
        # dimension is also chosen, points still need splitting into one
        # trace per shape category (Plotly assigns one symbol per trace),
        # but all traces must share the SAME color scale range (cmin/cmax)
        # so a given color means the same value everywhere on the plot --
        # otherwise each trace would auto-scale independently and the colors
        # would be incomparable across shapes.
        cmin, cmax = coords[beta_group_col].min(), coords[beta_group_col].max()
        group_iter = [(cat, coords[coords[shape_col] == cat]) for cat in shape_categories] if shape_col else [(None, coords)]
        for i, (shape_val, sub) in enumerate(group_iter):
            hover_text = [
                f"sample: {sid}<br>{_group_label(beta_group_col)}: {val:.1f}"
                + (f"<br>{_group_label(shape_col)}: {shape_val}" if shape_col else "")
                for sid, val in zip(sub.index, sub[beta_group_col])
            ]
            fig_pcoa.add_trace(go.Scatter(
                x=sub["Axis.1"], y=sub["Axis.2"], mode="markers",
                marker=dict(
                    size=13, color=sub[beta_group_col], colorscale="Viridis",
                    cmin=cmin, cmax=cmax,
                    showscale=(i == 0),  # only one colorbar, not one per trace
                    colorbar=dict(
                        title=dict(text=_group_label(beta_group_col), side="right"),
                        x=1.02, xanchor="left", len=0.8,
                    ) if i == 0 else None,
                    symbol=symbol_map.get(shape_val, "circle") if shape_col else "circle",
                    line=dict(width=1.5, color="rgba(255,255,255,0.6)"), opacity=0.9,
                ),
                text=hover_text, hoverinfo="text",
                name=str(shape_val) if shape_col else "",
                showlegend=False,
            ))

        # Shape gets its own legend (same pattern as the categorical branch);
        # color has no discrete legend here since it's the colorbar instead.
        for i, cat in enumerate(shape_categories):
            fig_pcoa.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=13, color="rgba(180,180,180,0.9)", symbol=symbol_map[cat]),
                name=str(cat), legendgroup="shape",
                legendgrouptitle_text=_group_label(shape_col) if i == 0 else None,
                showlegend=True,
            ))
    else:
        # Plotly Express merges color+symbol into one legend with a combined
        # entry per (color, shape) COMBINATION (e.g. "Season=Summer,
        # Position=Side"), which is clutter we want to avoid -- but the
        # earlier fix for that (separate color/shape legends built from
        # empty "dummy" placeholder traces) broke click-to-isolate, since
        # clicking a dummy entry toggles an empty trace with no real data
        # behind it. Fixed here by making each COLOR CATEGORY its own real
        # trace containing all of that category's actual points (with
        # shape still encoded via a per-point marker symbol array) -- so
        # the trace IS the legend entry, and clicking/double-clicking it
        # in the legend properly hides/isolates that category's real dots.
        color_categories = sorted(coords[beta_group_col].dropna().unique().tolist())
        palette = px.colors.qualitative.Plotly
        color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(color_categories)}

        for i, cat in enumerate(color_categories):
            sub = coords[coords[beta_group_col] == cat]
            symbols = [symbol_map.get(sv, "circle") for sv in sub[shape_col]] if shape_col else "circle"
            hover_text = [
                f"sample: {sid}<br>{_group_label(beta_group_col)}: {cat}"
                + (f"<br>{_group_label(shape_col)}: {sv}" if shape_col else "")
                for sid, sv in zip(sub.index, sub[shape_col] if shape_col else [None] * len(sub))
            ]
            fig_pcoa.add_trace(go.Scatter(
                x=sub["Axis.1"], y=sub["Axis.2"], mode="markers",
                marker=dict(
                    size=13, color=color_map[cat], symbol=symbols,
                    line=dict(width=1.5, color="rgba(255,255,255,0.6)"), opacity=0.9,
                ),
                text=hover_text, hoverinfo="text",
                name=str(cat), legendgroup="color",
                legendgrouptitle_text=_group_label(beta_group_col) if i == 0 else None,
                showlegend=True,
            ))

        # Shape legend stays reference-only (dummy entries, no real data
        # behind them): making BOTH color AND shape independently isolable
        # via the legend isn't achievable without either combinatorial
        # clutter (one entry per color x shape combination) or a trace per
        # shape category duplicated across colors -- out of scope here,
        # since color is the primary comparison variable being isolated.
        for i, cat in enumerate(shape_categories):
            fig_pcoa.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=13, color="rgba(180,180,180,0.9)", symbol=symbol_map[cat]),
                name=str(cat), legendgroup="shape",
                legendgrouptitle_text=_group_label(shape_col) if i == 0 else None,
                showlegend=True,
            ))

    # When a colorbar is present (continuous color), push the discrete
    # legend (shape categories) further right and add margin room so the
    # two titles don't overlap -- they previously both anchored near the
    # same default position and rendered stacked on top of each other.
    legend_x = 1.30 if beta_color_is_numeric else 1.02
    right_margin = 220 if beta_color_is_numeric else 120
    fig_pcoa.update_layout(
        height=550,
        xaxis_title=f"Axis 1 ({var_exp[0]*100:.1f}%)",
        yaxis_title=f"Axis 2 ({var_exp[1]*100:.1f}%)",
        legend=dict(groupclick="toggleitem", itemsizing="constant", x=legend_x, xanchor="left", y=1, yanchor="top"),
        margin=dict(r=right_margin),
    )
    st.plotly_chart(fig_pcoa, width="stretch")
    add_figure_download_buttons(fig_pcoa, filename=f"beta_diversity_pcoa_{beta_metric}_by_{beta_group_col}", key_prefix="beta_pcoa")

    if beta_color_is_numeric:
        st.caption(
            f"PERMANOVA isn't shown here since '{_group_label(beta_group_col)}' is a "
            f"continuous variable, not discrete groups -- switch 'Color PCoA by' to a "
            f"categorical variable (e.g. Season, Stage, Position) to test group separation."
        )
    elif st.button("Run PERMANOVA (999 permutations)"):
        grouping = wide_df.set_index("sample.id")[beta_group_col]
        with st.spinner("Permuting..."):
            result = div.permanova(dist, grouping, permutations=999)
        st.write(
            f"**Pseudo-F = {result['F']:.3f}, R\u00b2 = {result['R2']:.3f}, p = {result['p_value']:.4f}** "
            f"({result['permutations']} permutations) -- tests whether "
            f"'{_group_label(beta_group_col)}' group centroids differ in {beta_metric} space."
        )

    st.divider()
    st.subheader("PERMANOVA table")
    st.caption(
        "Runs the omnibus PERMANOVA for each selected grouping variable on the same "
        f"{beta_metric} distance matrix ({rank} rank) and lays the results out as one "
        "row per variable -- the standard table format for a paper, rather than "
        "checking one variable at a time above."
    )

    # Carcass is excluded from the default selection (still pickable) since
    # it's an identifier with many levels/small per-group n, not usually a
    # meaningful PERMANOVA factor on its own.
    permanova_table_candidates = [c for c in meta_grouping_cols if c in wide_df.columns]
    selected_vars = st.multiselect(
        "Grouping variables to test",
        options=permanova_table_candidates,
        default=[c for c in permanova_table_candidates if c != "Carcass"],
        format_func=_group_label,
    )

    if selected_vars and st.button("Run PERMANOVA table", type="primary"):
        dist_by_group = {
            _group_label(v): (dist, wide_df.set_index("sample.id")[v]) for v in selected_vars
        }
        with st.spinner(f"Running PERMANOVA for {len(selected_vars)} grouping variable(s)..."):
            table = div.permanova_table(dist_by_group, permutations=999)
        st.session_state["permanova_table"] = table
        st.session_state["permanova_table_rank"] = rank
        st.session_state["permanova_table_metric"] = beta_metric
        st.session_state["permanova_table_vars"] = selected_vars

    if "permanova_table" in st.session_state:
        table = st.session_state["permanova_table"]
        st.caption(
            f"{st.session_state['permanova_table_rank']} rank, "
            f"{st.session_state['permanova_table_metric']} distance"
        )
        display_table = table.copy()
        display_table["pseudo_F"] = display_table["pseudo_F"].round(2)
        display_table["R2"] = display_table["R2"].round(3)
        display_table["p_value"] = display_table["p_value"].round(4)
        st.dataframe(display_table, width="stretch")
        st.download_button(
            "Download PERMANOVA table (CSV)",
            data=table.to_csv().encode("utf-8"),
            file_name=f"permanova_table_{rank}_{beta_metric}.csv",
            mime="text/csv",
            key="permanova_table_csv",
        )

        st.markdown("**Pairwise post-hoc** (which specific groups differ, BH-corrected)")
        pairwise_var = st.selectbox(
            "Run pairwise PERMANOVA for",
            options=st.session_state["permanova_table_vars"],
            format_func=_group_label,
            key="pairwise_var_choice",
        )
        if st.button("Run pairwise PERMANOVA (999 permutations)"):
            pairwise_grouping = wide_df.set_index("sample.id")[pairwise_var]
            with st.spinner(f"Running pairwise PERMANOVA for {_group_label(pairwise_var)}..."):
                pairwise_table = div.pairwise_permanova(dist, pairwise_grouping, permutations=999)
            st.session_state["pairwise_table"] = pairwise_table
            st.session_state["pairwise_var_used"] = pairwise_var

        if "pairwise_table" in st.session_state:
            pt = st.session_state["pairwise_table"].copy()
            if pt.empty:
                st.info(f"'{_group_label(st.session_state['pairwise_var_used'])}' has fewer than 2 groups -- nothing to compare pairwise.")
            else:
                st.caption(f"Pairwise PERMANOVA -- {_group_label(st.session_state['pairwise_var_used'])}")
                pt["pseudo_F"] = pt["pseudo_F"].round(2)
                pt["R2"] = pt["R2"].round(3)
                pt["p_value"] = pt["p_value"].round(4)
                pt["q_value"] = pt["q_value"].round(4)
                st.dataframe(pt, width="stretch")
                st.download_button(
                    "Download pairwise PERMANOVA (CSV)",
                    data=st.session_state["pairwise_table"].to_csv(index=False).encode("utf-8"),
                    file_name=f"pairwise_permanova_{rank}_{beta_metric}_{st.session_state['pairwise_var_used']}.csv",
                    mime="text/csv",
                    key="pairwise_table_csv",
                )
    st.divider()
    st.subheader("Two-way PERMANOVA (with interaction)")
    st.caption(
        "Tests whether two grouping variables INTERACT -- e.g. whether the Stage effect "
        "on community composition looks different depending on Season, not just whether "
        "Season and Stage each separately matter. Equivalent to "
        "`adonis2(dist ~ A * B, by = \"terms\")` in R's vegan package."
    )
    interaction_candidates = [c for c in meta_grouping_cols if c in wide_df.columns]
    int_col1, int_col2 = st.columns(2)
    with int_col1:
        factor_a = st.selectbox("Factor A", options=interaction_candidates, key="factor_a",
                                 format_func=_group_label)
    with int_col2:
        factor_b_options = [c for c in interaction_candidates if c != factor_a]
        factor_b = st.selectbox("Factor B", options=factor_b_options, key="factor_b",
                                 format_func=_group_label)

    if st.button("Run two-way PERMANOVA", type="primary"):
        grouping_a = wide_df.set_index("sample.id")[factor_a]
        grouping_b = wide_df.set_index("sample.id")[factor_b]
        with st.spinner(f"Running two-way PERMANOVA ({_group_label(factor_a)} * {_group_label(factor_b)})..."):
            factorial_table = div.permanova_factorial(
                dist, {factor_a: grouping_a, factor_b: grouping_b}, interactions=True, permutations=999,
            )
        st.session_state["factorial_table"] = factorial_table
        st.session_state["factorial_factors"] = (factor_a, factor_b)
        st.session_state["factorial_rank"] = rank
        st.session_state["factorial_metric"] = beta_metric

    if "factorial_table" in st.session_state:
        fa, fb = st.session_state["factorial_factors"]
        st.caption(
            f"{st.session_state['factorial_rank']} rank, {st.session_state['factorial_metric']} distance -- "
            f"{_group_label(fa)} * {_group_label(fb)}"
        )
        display_factorial = st.session_state["factorial_table"].copy()
        for col in ["SS", "R2", "pseudo_F"]:
            display_factorial[col] = display_factorial[col].round(3)
        display_factorial["p_value"] = display_factorial["p_value"].round(4)
        st.dataframe(display_factorial, width="stretch")
        st.caption(
            f"The '{fa}:{fb}' row is the interaction -- if its p-value clears your threshold, "
            f"the effect of one variable genuinely depends on the level of the other, and "
            f"reporting the two main effects on their own would be misleading."
        )
        st.download_button(
            "Download two-way PERMANOVA table (CSV)",
            data=st.session_state["factorial_table"].to_csv().encode("utf-8"),
            file_name=f"permanova_factorial_{rank}_{beta_metric}_{fa}_x_{fb}.csv",
            mime="text/csv",
            key="factorial_table_csv",
        )
else:
    st.warning("Not enough samples with count columns to compute beta diversity.")