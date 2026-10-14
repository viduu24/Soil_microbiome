# pages/8_Predict_PMI_ADH.py
"""
Predict PMI / ADH from soil microbiome composition, validated honestly.

With only ~17 carcasses, ordinary random cross-validation would be badly
over-optimistic: Side and Under samples taken from the same carcass on the
same day are near-duplicates, so a random split lets the model "recognise the
carcass" instead of learning decomposition. Here every fold holds out WHOLE
carcasses (leave-one-carcass-out by default).

Models are compared against two baselines, following the design of Maltecca
et al. 2019 (Sci Rep 9:6574):
  * mean-only          -- predicts the training average (no information)
  * metadata-only      -- Season + Position, no taxa  (the "null model")
and four taxa-based models: Random Forest, Gradient Boosting, Ridge on
CLR-transformed abundances, and kernel ridge on a Jensen-Shannon kernel,
plus the average of those four.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
PROCESSED_DIR = SRC_DIR / "processed"

from data_source import data_available, get_wide
from export_utils import add_figure_download_buttons

st.set_page_config(page_title="Predict PMI / ADH", layout="wide")
st.title("Predict PMI and ADH from the soil microbiome")
st.caption(
    "Every cross-validation fold holds out whole carcasses, so the scores reflect how well the "
    "model would do on a carcass it has never seen. Compare each model to the two baselines: "
    "if the taxa models don't clearly beat them, the microbiome isn't adding information."
)

if not data_available(PROCESSED_DIR):
    st.warning("No processed data found. Run `wrangle.py` first, or use the Upload Data page.")
    st.stop()

POSITION_COL = "Family"          # Side / Under (not the taxonomic rank)
SEASON_COL = "Season"
MEAN_NAME = "Mean only (no information)"
META_NAME = "Metadata only (Season + Position)"
ENSEMBLE_NAME = "Average of taxa models"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
rank = st.selectbox("Taxonomic rank", ["Family", "Phylum"], key="ml_rank")
wide = get_wide(PROCESSED_DIR, rank).reset_index(drop=True)

relabund_cols = [c for c in wide.columns if c.endswith("_relabund")]
if relabund_cols:
    feature_cols = relabund_cols
    taxon_labels = [c.replace("_relabund", "") for c in relabund_cols]
else:
    feature_cols = [c for c in wide.columns if c not in ("sample.id",) and pd.api.types.is_numeric_dtype(wide[c])]
    taxon_labels = list(feature_cols)
rel_all = wide[feature_cols].copy()
rel_all.columns = taxon_labels
rel_all = rel_all.apply(pd.to_numeric, errors="coerce").fillna(0.0)

feature_like = set(feature_cols)
numeric_cols = [c for c in wide.columns
                if c not in feature_like and c != "sample.id" and pd.api.types.is_numeric_dtype(wide[c])]
preferred = [c for c in ["ADH", "DaysSinceDeath", "PMI", "PMSI", "Days"] if c in numeric_cols]
target_options = preferred + [c for c in numeric_cols if c not in preferred]
if not target_options:
    st.error("No numeric outcome columns (e.g. ADH, DaysSinceDeath) found in the processed data.")
    st.stop()

group_options = [c for c in wide.columns if c not in feature_like and c != "sample.id"]
group_default = group_options.index("Carcass") if "Carcass" in group_options else 0
has_position = POSITION_COL in wide.columns
has_season = SEASON_COL in wide.columns

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    target = st.selectbox("Predict", target_options, key="ml_target",
                          help="PMI is usually the days-since-death column; ADH is accumulated degree hours.")
with c2:
    group_col = st.selectbox("Keep together in CV (never split)", group_options, index=group_default,
                             key="ml_group", help="Should be the carcass ID.")
with c3:
    if has_position:
        positions = sorted(wide[POSITION_COL].dropna().astype(str).unique().tolist())
        pos_scope = st.selectbox("Sampling position", ["All"] + positions, key="ml_pos")
    else:
        pos_scope = "All"

d1, d2, d3, d4 = st.columns(4)
with d1:
    cv_choice = st.radio("Validation", ["Leave one carcass out", "Grouped k-fold"], key="ml_cv")
    k_folds = st.number_input("k (grouped k-fold)", 2, 20, 5, key="ml_k", disabled=cv_choice.startswith("Leave"))
with d2:
    prevalence = st.slider("Keep taxa present in at least (% of samples)", 0, 50, 10, 5, key="ml_prev")
with d3:
    add_meta = st.checkbox("Also give taxa models Season + Position as features", False, key="ml_addmeta")
with d4:
    use_log = st.checkbox("Log-transform the outcome", False, key="ml_log",
                          help="Helps if the outcome is strongly right-skewed. Scores are always reported on the original scale.")

with st.expander("Model settings"):
    s1, s2, s3, s4 = st.columns(4)
    rf_trees = s1.number_input("Random Forest trees", 100, 2000, 500, 100, key="ml_rf")
    gbm_trees = s2.number_input("Gradient Boosting trees", 50, 1000, 300, 50, key="ml_gbm")
    ridge_alpha = s3.number_input("Ridge alpha", 0.01, 10000.0, 100.0, key="ml_ridge")
    kernel_alpha = s4.number_input("Kernel ridge alpha", 0.001, 100.0, 0.1, key="ml_kalpha")

# ---------------------------------------------------------------------------
# Subset
# ---------------------------------------------------------------------------
mask = wide[target].notna() & wide[group_col].notna()
if pos_scope != "All":
    mask &= wide[POSITION_COL].astype(str) == pos_scope
sub = wide.loc[mask].reset_index(drop=True)
rel = rel_all.loc[mask].reset_index(drop=True)
rel = rel.loc[:, (rel > 0).mean() >= prevalence / 100.0]
rel = rel.loc[:, rel.sum() > 0]

n_samples = len(sub)
n_groups = sub[group_col].nunique()
m1, m2, m3 = st.columns(3)
m1.metric("Samples", n_samples)
m2.metric("Carcasses (groups)", n_groups)
m3.metric("Taxa used", rel.shape[1])

if n_groups < 3:
    st.error("Need at least 3 carcasses to cross-validate by carcass.")
    st.stop()
if rel.shape[1] < 2:
    st.error("Fewer than 2 taxa left after the prevalence filter -- lower the threshold.")
    st.stop()
if use_log and (sub[target] < 0).any():
    st.error("Outcome has negative values, so it can't be log-transformed.")
    st.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clr(x: np.ndarray, pseudo: float = 1e-4) -> np.ndarray:
    x = x + pseudo
    x = x / x.sum(axis=1, keepdims=True)
    lx = np.log(x)
    return lx - lx.mean(axis=1, keepdims=True)


def _jsd_kernel(x: np.ndarray) -> np.ndarray:
    """Gaussian kernel on Jensen-Shannon distance between samples."""
    p = x + 1e-12
    p = p / p.sum(axis=1, keepdims=True)
    d = np.nan_to_num(squareform(pdist(p, metric="jensenshannon")))
    off = d[np.triu_indices_from(d, k=1)]
    sigma = float(np.median(off[off > 0])) if (off > 0).any() else 1.0
    return np.exp(-(d ** 2) / (2 * sigma ** 2))


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    ok = np.std(y_pred) > 0 and np.std(y_true) > 0
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - np.sum(err ** 2) / sst) if sst > 0 else np.nan,
        "Pearson r": float(pearsonr(y_true, y_pred)[0]) if ok else np.nan,
        "Spearman": float(spearmanr(y_true, y_pred)[0]) if ok else np.nan,
    }


def _run() -> dict:
    y_raw = sub[target].to_numpy(float)
    y = np.log1p(y_raw) if use_log else y_raw
    groups = sub[group_col].astype(str).to_numpy()
    n = len(sub)

    x_rel = rel.to_numpy(float)
    x_clr = _clr(x_rel)
    meta_cols = [c for c in (SEASON_COL, POSITION_COL) if c in sub.columns]
    if meta_cols:
        meta_df = pd.get_dummies(sub[meta_cols].fillna("NA").astype(str))
        x_meta = meta_df.to_numpy(float)
        meta_names = list(meta_df.columns)
    else:
        x_meta, meta_names = np.zeros((n, 0)), []
    x_rf = np.hstack([x_rel, x_meta]) if add_meta else x_rel
    x_lin = np.hstack([x_clr, x_meta]) if add_meta else x_clr
    kern = _jsd_kernel(x_rel)

    def mean_fn(tr, te):
        return np.full(len(te), y[tr].mean())

    def meta_fn(tr, te):
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)
        m.fit(x_meta[tr], y[tr])
        return m.predict(x_meta[te])

    def rf_fn(tr, te):
        m = RandomForestRegressor(n_estimators=int(rf_trees), min_samples_leaf=2, max_features="sqrt",
                                  random_state=0, n_jobs=-1)
        m.fit(x_rf[tr], y[tr])
        return m.predict(x_rf[te])

    def gbm_fn(tr, te):
        m = GradientBoostingRegressor(n_estimators=int(gbm_trees), max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=0)
        m.fit(x_rf[tr], y[tr])
        return m.predict(x_rf[te])

    def ridge_fn(tr, te):
        m = make_pipeline(StandardScaler(), Ridge(alpha=float(ridge_alpha)))
        m.fit(x_lin[tr], y[tr])
        return m.predict(x_lin[te])

    def kernel_fn(tr, te):
        mu = y[tr].mean()
        m = KernelRidge(alpha=float(kernel_alpha), kernel="precomputed")
        m.fit(kern[np.ix_(tr, tr)], y[tr] - mu)
        return m.predict(kern[np.ix_(te, tr)]) + mu

    models = {MEAN_NAME: mean_fn}
    if x_meta.shape[1]:
        models[META_NAME] = meta_fn
    models.update({
        "Random Forest": rf_fn,
        "Gradient Boosting": gbm_fn,
        "Ridge (CLR)": ridge_fn,
        "Kernel ridge (Jensen-Shannon)": kernel_fn,
    })
    taxa_models = ["Random Forest", "Gradient Boosting", "Ridge (CLR)", "Kernel ridge (Jensen-Shannon)"]

    splitter = LeaveOneGroupOut() if cv_choice.startswith("Leave") else GroupKFold(
        n_splits=int(min(k_folds, len(set(groups)))))
    splits = list(splitter.split(x_rel, y, groups))

    oof = {name: np.full(n, np.nan) for name in models}
    bar = st.progress(0.0, text="Cross-validating...")
    for i, (tr, te) in enumerate(splits):
        for name, fn in models.items():
            oof[name][te] = fn(tr, te)
        bar.progress((i + 1) / len(splits), text=f"Fold {i + 1} of {len(splits)}")
    bar.empty()

    preds = pd.DataFrame(oof)
    preds[ENSEMBLE_NAME] = preds[taxa_models].mean(axis=1)
    if use_log:
        preds = np.expm1(preds)

    rows = [{"Model": name, **_score(y_raw, preds[name].to_numpy())} for name in preds.columns]
    scores = pd.DataFrame(rows)
    base_mae = float(scores.loc[scores["Model"] == MEAN_NAME, "MAE"].iloc[0])
    scores["MAE improvement vs mean-only"] = 1 - scores["MAE"] / base_mae

    imp_model = RandomForestRegressor(n_estimators=int(rf_trees), min_samples_leaf=2, max_features="sqrt",
                                      random_state=0, n_jobs=-1).fit(x_rf, y)
    names = list(rel.columns) + (meta_names if add_meta else [])
    importance = (pd.Series(imp_model.feature_importances_, index=names)
                  .sort_values(ascending=False).head(15).rename("importance").reset_index()
                  .rename(columns={"index": "feature"}))

    detail = sub[[c for c in {group_col, SEASON_COL, POSITION_COL} if c in sub.columns]].copy()
    detail.insert(0, "observed", y_raw)
    return {
        "preds": preds, "detail": detail, "scores": scores, "importance": importance,
        "group_col": group_col, "target": target, "n_folds": len(splits),
        "settings": (f"{target} | {rank} | position={pos_scope} | {cv_choice} | "
                     f"{n_samples} samples, {n_groups} carcasses, {rel.shape[1]} taxa"
                     f"{' | log outcome' if use_log else ''}{' | +Season/Position' if add_meta else ''}"),
    }


# ---------------------------------------------------------------------------
# Run + results
# ---------------------------------------------------------------------------
if st.button("Run benchmark", type="primary", key="ml_run_btn"):
    st.session_state["ml_result"] = _run()

res = st.session_state.get("ml_result")
if res is None:
    st.info("Choose settings above, then click **Run benchmark**.")
    st.stop()

st.subheader("Results")
st.caption(f"Settings: {res['settings']}  ({res['n_folds']} folds)")

scores = res["scores"]
st.dataframe(
    scores.round({"MAE": 2, "RMSE": 2, "R2": 3, "Pearson r": 3, "Spearman": 3, "MAE improvement vs mean-only": 3}),
    hide_index=True, use_container_width=True,
)
st.caption(
    "MAE/RMSE are in the outcome's own units (lower is better). R2 below 0 means worse than always "
    "guessing the overall average. 'MAE improvement' is the fraction of the mean-only error removed."
)

taxa_scores = scores[~scores["Model"].isin([MEAN_NAME, META_NAME])]
best = taxa_scores.sort_values("MAE").iloc[0]["Model"] if len(taxa_scores) else ENSEMBLE_NAME
pred_cols = list(res["preds"].columns)
model_choice = st.selectbox("Show details for", pred_cols, index=pred_cols.index(best), key="ml_show")

detail = res["detail"].copy()
detail["predicted"] = res["preds"][model_choice].to_numpy()
detail["abs_error"] = (detail["predicted"] - detail["observed"]).abs()
color_opts = [c for c in [POSITION_COL, SEASON_COL, res["group_col"]] if c in detail.columns]
color_by = st.selectbox("Color points by", color_opts, key="ml_color") if color_opts else None

left, right = st.columns(2)
with left:
    fig = px.scatter(detail, x="observed", y="predicted", color=color_by,
                     hover_data=[c for c in detail.columns if c not in ("observed", "predicted")],
                     title=f"{res['target']}: observed vs predicted ({model_choice}, held-out carcasses)")
    lo = float(min(detail["observed"].min(), detail["predicted"].min()))
    hi = float(max(detail["observed"].max(), detail["predicted"].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect prediction",
                             line=dict(dash="dash", color="gray")))
    fig.update_layout(xaxis_title=f"Observed {res['target']}", yaxis_title=f"Predicted {res['target']}")
    st.plotly_chart(fig, use_container_width=True, key="ml_scatter")
    add_figure_download_buttons(fig, filename=f"predict_{res['target']}_{model_choice.split()[0].lower()}",
                                key_prefix="ml_scatter_dl")
with right:
    per_group = (detail.groupby(res["group_col"])["abs_error"].mean().sort_values(ascending=False)
                 .rename("MAE").reset_index())
    fig2 = px.bar(per_group, x=res["group_col"], y="MAE",
                  title=f"Error by held-out carcass ({model_choice})")
    fig2.update_layout(xaxis_title="Carcass", yaxis_title=f"MAE ({res['target']})")
    st.plotly_chart(fig2, use_container_width=True, key="ml_bycarcass")
    st.caption("A few carcasses with much larger error often point to a biologically different group "
               "(e.g. the Summer stillborn vs adult tracks) rather than a modeling problem.")

with st.expander("Top taxa by Random Forest importance (fit on all samples, for interpretation only)"):
    fig3 = px.bar(res["importance"].iloc[::-1], x="importance", y="feature", orientation="h")
    st.plotly_chart(fig3, use_container_width=True, key="ml_importance")

out = detail.copy()
for c in pred_cols:
    out[f"pred_{c}"] = res["preds"][c].to_numpy()
st.download_button("Download held-out predictions (CSV)", out.to_csv(index=False).encode("utf-8"),
                   file_name=f"heldout_predictions_{res['target']}.csv", mime="text/csv", key="ml_csv")

with st.expander("How to read this / cautions"):
    st.markdown(
        "- **Stage is deliberately not used as a predictor**: it is defined from how decomposed the carcass "
        "looks, so it would leak the answer for PMI/ADH.\n"
        "- **Season and Position are legitimate inputs** (a case investigator knows them), which is why the "
        "metadata-only baseline exists. A taxa model only 'earns' its place if it beats that baseline.\n"
        "- With ~17 carcasses the estimates are noisy: one odd carcass can move every score. "
        "Look at the per-carcass error plot, not just the summary table.\n"
        "- All prediction is on held-out carcasses. Never tune settings by repeatedly re-running until the "
        "scores look good; pick settings first, then report.\n"
        "- Following Maltecca et al. 2019, averaging across model types is usually a safer strategy than "
        "picking a single winner."
    )