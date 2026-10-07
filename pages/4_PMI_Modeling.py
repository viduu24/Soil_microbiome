# src/pages/3_PMI_Modeling.py
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
PROCESSED_DIR = SRC_DIR / "processed"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import modeling as mdl
import data_source as ds
from export_utils import add_figure_download_buttons

st.set_page_config(page_title="Predictive Modeling", layout="wide")
st.title("Predictive Modeling: PMI / Stage / Position / Provenance")
st.caption(
    "Random forest modeling of microbial composition against a chosen "
    "target -- automatically runs as REGRESSION for a numeric target (e.g. ADH, "
    "DaysSinceDeath) or CLASSIFICATION for a categorical target (e.g. Stage, sampling "
    "Position, Season, or Carcass -- the latter is a soil-provenance question: can "
    "microbial composition identify which specific site a sample came from?), plus "
    "quadratic succession curves for individual taxa against a numeric time axis. Mirrors "
    "the random forest + quadratic-regression modeling in Kaszubinski et al. 2022, which "
    "built separate models for both continuous (ADD) and categorical (collection date) targets."
)


def load_long():
    return ds.get_combined_long(PROCESSED_DIR)


if not ds.data_available(PROCESSED_DIR):
    st.error("No data available -- run wrangle.py locally, or use the Upload Data page.")
    st.stop()

df_all = load_long()
meta_cols = [c for c in df_all.columns if c not in ("taxon", "rel_abund", "rank")]

# "Family" in the metadata means sampling position (Side vs. Under the carcass),
# not the Family taxonomic rank chosen below -- label it clearly in the UI.
TARGET_LABELS = {"Family": "Position (Side/Under carcass)"}
def _target_label(c: str) -> str:
    return TARGET_LABELS.get(c, c)

# Columns that aren't meaningful PREDICTION targets: the sample identifier
# itself, calendar dates (not a model-friendly dtype), Order (constant --
# always "Soil" after wrangle.py's filter, same reasoning as the Diversity
# and Differential Abundance pages), and the daily weather covariates
# (Max/Min/Mean Air Temp, Precipitation, Humidity, Wind Speed). Weather is
# excluded as a PREDICTION TARGET because the causality runs the wrong way
# for a forensic question: weather influences which bacteria thrive, not
# the other way around -- "can microbial abundance predict what the
# temperature was" isn't a meaningful thing to ask here. Carcass IS
# included as a target, despite being an identifier: predicting which
# specific carcass/site a sample came from purely from its microbial
# composition is a genuine forensic soil-provenance question (can the
# microbiome "fingerprint" identify a location?), not just an ID-guessing
# exercise.
WEATHER_COVARIATES = {
    "Max Air Temp", "Min Air Temp", "Mean Air Temp", "Precipitation",
    "Max Relative Humidity", "Min Relative Humidity", "Mean Relative Humidity",
    "Max Wind Speed",
}
EXCLUDED_TARGETS = {"sample.id"} | WEATHER_COVARIATES
# Any column that's constant across every sample (a single unique value)
# isn't a meaningful prediction target -- e.g. our own "Order" column is
# always "Soil" after wrangle.py's filter, but an uploaded dataset (see
# pages/0_Upload_Data.py) could use a differently-named filter column that
# ends up just as constant post-filtering. Detecting this generically
# (rather than hardcoding "Order") means the exclusion works regardless
# of what a researcher's own filter column happens to be called.
constant_cols = {c for c in meta_cols if df_all[c].nunique(dropna=True) <= 1}
EXCLUDED_TARGETS = EXCLUDED_TARGETS | constant_cols
target_candidates = [
    c for c in meta_cols
    if c not in EXCLUDED_TARGETS and not pd.api.types.is_datetime64_any_dtype(df_all[c])
]
# Put the most forensically relevant targets first; anything else still
# appears, just further down the list.
preferred_order = ["Stage", "ADH", "DaysSinceDeath", "Season", "Family"]
target_candidates = sorted(
    target_candidates,
    key=lambda c: (preferred_order.index(c) if c in preferred_order else len(preferred_order), c),
)
numeric_targets = [c for c in target_candidates if pd.api.types.is_numeric_dtype(df_all[c])]

# Separate, WIDER numeric list for the quadratic trajectory tab's x-axis
# (tab2 below): weather covariates ARE valid here, since "does taxon
# abundance track with temperature/humidity/etc." is the causally sensible
# direction (weather -> taxa), even though the reverse isn't a valid
# prediction target above.
trajectory_x_candidates = numeric_targets + sorted(
    c for c in WEATHER_COVARIATES if c in meta_cols
)

st.sidebar.header("Data")
rank = st.sidebar.radio("Taxonomic rank", options=["Family", "Phylum"], horizontal=True)
target_col = st.sidebar.selectbox(
    "Target variable", options=target_candidates or ["ADH"], format_func=_target_label,
    help="Numeric targets (ADH, DaysSinceDeath, ...) run as regression; "
         "categorical targets (Stage, Position, Season, ...) run as classification.",
)

# "Family" in the metadata means sampling position (Side vs. Under the carcass),
# not the Family taxonomic rank chosen above. Offer it as a subset filter here,
# mirroring the papers' separate "internal-only" / "external-only" random
# forest models (Table 3, Kaszubinski et al. 2022) rather than a full grouping
# variable, since this page predicts a continuous target rather than comparing
# groups directly.
position_col = "Family" if "Family" in meta_cols else None
position_subset = "All samples"
if position_col:
    position_lookup = df_all.drop_duplicates("sample.id").set_index("sample.id")[position_col]
    position_options = ["All samples"] + sorted(position_lookup.dropna().unique().tolist())
    position_subset = st.sidebar.radio(
        "Sample position subset", options=position_options, horizontal=True,
        help="Restrict the model to soil sampled directly under the carcass, "
             "beside it, or use all samples together.",
    )

wide = mdl.pivot_relabund_wide(df_all, rank=rank)
target_lookup = df_all[df_all["rank"] == rank].drop_duplicates("sample.id").set_index("sample.id")[target_col]

if position_col and position_subset != "All samples":
    keep_ids = position_lookup[position_lookup == position_subset].index
    wide = wide.loc[wide.index.intersection(keep_ids)]
    target_lookup = target_lookup.loc[target_lookup.index.intersection(keep_ids)]
    st.info(f"Restricted to **{position_subset}** samples only ({len(wide)} samples).")

tab1, tab2 = st.tabs(["Random forest regression", "Quadratic taxon trajectories"])

with tab1:
    st.subheader(f"Random forest: predicting {_target_label(target_col)} from {rank}-level relative abundance")
    n_trees = st.slider("Number of trees", 100, 1000, 500, step=100)
    n_splits = st.slider("Cross-validation folds", 3, 10, 5)

    if st.button("Fit random forest model", type="primary"):
        with st.spinner("Fitting (with k-fold cross-validated predictions)..."):
            result = mdl.fit_random_forest(wide, target_lookup, n_trees=n_trees, n_splits=n_splits)
        st.session_state["rf_result"] = result
        st.session_state["rf_target"] = target_col
        st.session_state["rf_rank"] = rank

    if "rf_result" in st.session_state and st.session_state.get("rf_target") == target_col:
        result = st.session_state["rf_result"]
        target_label = _target_label(st.session_state["rf_target"])

        if result["task"] == "regression":
            st.caption("Detected as a **numeric** target -> ran as regression.")
            col1, col2, col3 = st.columns(3)
            col1.metric("Cross-validated R²", f"{result['r2']:.3f}")
            col2.metric("Mean squared error", f"{result['mse']:.1f}")
            col3.metric("n samples", result["n_samples"])

            preds = result["predictions"]
            fig = px.scatter(
                preds, x="true", y="predicted",
                labels={"true": f"True {target_label}", "predicted": f"Predicted {target_label}"},
            )
            min_v, max_v = preds[["true", "predicted"]].min().min(), preds[["true", "predicted"]].max().max()
            fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", line=dict(dash="dash", color="gray"), name="1:1"))
            fig.update_layout(height=500)
            st.plotly_chart(fig, width="stretch")
            add_figure_download_buttons(fig, filename=f"rf_regression_{target_col}_{rank}", key_prefix="rf_reg")

        else:  # classification
            st.caption(f"Detected as a **categorical** target ({len(result['labels'])} classes) -> ran as classification.")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Cross-validated accuracy", f"{result['accuracy']:.3f}")
            col2.metric("Balanced accuracy", f"{result['balanced_accuracy']:.3f}")
            col3.metric("n samples", result["n_samples"])
            col4.metric("CV folds used", result["n_splits_used"])
            st.caption(
                "Balanced accuracy = average recall across classes, treating each class "
                "equally regardless of size -- a fairer summary than raw accuracy when "
                "classes are imbalanced (e.g. Fresh with 90 samples vs. Active with 12). "
                "The model was also trained with class_weight='balanced' so it doesn't "
                "implicitly favor the largest class."
            )

            labels = result["labels"]
            cm = result["confusion_matrix"]
            fig_cm = px.imshow(
                cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x=f"Predicted {target_label}", y=f"True {target_label}", color="Count"),
            )
            fig_cm.update_layout(height=450)
            st.plotly_chart(fig_cm, width="stretch")
            add_figure_download_buttons(fig_cm, filename=f"rf_confusion_matrix_{target_col}_{rank}", key_prefix="rf_cm")
            st.caption(
                "Rows = true class, columns = predicted class. Diagonal cells are correct "
                "predictions; off-diagonal cells show which classes get confused with which."
            )

            report_df = pd.DataFrame(result["classification_report"]).T
            report_df = report_df.loc[[l for l in labels if l in report_df.index]]
            st.dataframe(
                report_df[["precision", "recall", "f1-score", "support"]].round(3),
                width="stretch",
            )

        st.subheader("Top predictor taxa (feature importance)")
        importance = result["feature_importance"].head(15).sort_values()
        fig_imp = px.bar(importance, orientation="h", labels={"value": "Feature importance", "index": "Taxon"})
        fig_imp.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_imp, width="stretch")
        add_figure_download_buttons(fig_imp, filename=f"rf_feature_importance_{target_col}_{rank}", key_prefix="rf_imp")
    else:
        st.info("Click **Fit random forest model** to train and cross-validate.")

with tab2:
    st.subheader(f"Quadratic succession curve for a single taxon ({rank} rank)")
    # Independent from the main "Target variable" selector above (which can
    # now be categorical, e.g. Stage or Position) -- quadratic curve fitting
    # needs a continuous x-axis, so this only offers numeric columns.
    traj_x_col = st.selectbox(
        "X-axis for trajectory", options=trajectory_x_candidates or ["ADH"], format_func=_target_label,
        help="Quadratic curve fitting needs a continuous axis, so only numeric "
             "columns (ADH, DaysSinceDeath, ...) are offered here.",
    )
    traj_x_lookup = df_all[df_all["rank"] == rank].drop_duplicates("sample.id").set_index("sample.id")[traj_x_col]
    if position_col and position_subset != "All samples":
        traj_x_lookup = traj_x_lookup.loc[traj_x_lookup.index.intersection(keep_ids)]

    taxon_options = sorted(wide.columns.tolist())
    taxon = st.selectbox("Taxon", options=taxon_options)

    x = traj_x_lookup.reindex(wide.index)
    y = wide[taxon]
    traj = mdl.quadratic_trajectory(x, y)

    if traj is None:
        st.warning(
            f"Not enough valid (non-missing) data points to fit a quadratic curve for "
            f"'{taxon}' against {_target_label(traj_x_col)} (need at least 4). Try a "
            f"different taxon, x-axis variable, or widen the sample position subset."
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Observed", marker=dict(color="steelblue")))
        fig.add_trace(go.Scatter(
            x=np.concatenate([traj["x_grid"], traj["x_grid"][::-1]]),
            y=np.concatenate([traj["y_upper"], traj["y_lower"][::-1]]),
            fill="toself", fillcolor="rgba(200,30,30,0.15)", line=dict(color="rgba(255,255,255,0)"),
            name="95% CI", showlegend=True,
        ))
        fig.add_trace(go.Scatter(x=traj["x_grid"], y=traj["y_fit"], mode="lines", name="Quadratic fit", line=dict(color="firebrick")))
        fig.update_layout(
            xaxis_title=_target_label(traj_x_col), yaxis_title=f"{taxon} relative abundance",
            height=500,
        )
        st.plotly_chart(fig, width="stretch")
        safe_taxon = str(taxon).replace(" ", "_").replace("/", "-")
        add_figure_download_buttons(
            fig, filename=f"trajectory_{safe_taxon}_vs_{traj_x_col}_{rank}", key_prefix="quad_traj",
        )
        st.caption(f"Quadratic fit R² = {traj['r2']:.3f} (n = {traj['n']})")