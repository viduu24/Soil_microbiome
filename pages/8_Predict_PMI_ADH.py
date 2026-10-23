# pages/8_Predict_PMI_ADH.py
"""
Predict ADH / days since death (numbers) or Position / Stage / Season
(categories) from soil microbiome composition, validated honestly.

With only ~17 carcasses, ordinary random cross-validation would be badly
over-optimistic: Side and Under samples from the same carcass on the same day
are near-duplicates, so a random split lets the model "recognise the carcass"
instead of learning decomposition. Here every fold holds out WHOLE carcasses
(leave-one-carcass-out by default).

Optional FINAL TEST SET: reserve a few carcasses before any model comparison.
Cross-validation (model comparison) uses only the remaining carcasses; each
model is then trained on all of those and scored once on the reserved carcasses.

Numeric outcomes -- baselines: mean-only, metadata-only (Season + Position).
Category outcomes -- baselines: majority-class, metadata-only (Season and/or
Position, whichever is not the thing being predicted).
Taxa models: Random Forest, Gradient Boosting, Ridge/Logistic on CLR-transformed
abundances, a Jensen-Shannon kernel model (kernel ridge / SVM), and an average.
Design follows Maltecca et al. 2019 (Sci Rep 9:6574).
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
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
PROCESSED_DIR = SRC_DIR / "processed"

from data_source import data_available, get_wide
from export_utils import add_figure_download_buttons

st.set_page_config(page_title="Predict PMI / ADH / Position / Stage", layout="wide")
st.title("Predict ADH, days since death, Position, Stage and Season from the soil microbiome")
st.caption(
    "Every cross-validation fold holds out whole carcasses, so the scores reflect how well the "
    "model would do on a carcass it has never seen. Compare each model to the baselines: "
    "if the taxa models don't clearly beat them, the microbiome isn't adding information."
)

if not data_available(PROCESSED_DIR):
    st.warning("No processed data found. Run `wrangle.py` first, or use the Upload Data page.")
    st.stop()

POSITION_COL = "Family"          # Side / Under (not the taxonomic rank)
SEASON_COL = "Season"
STAGE_COL = "Stage"
MEAN_NAME = "Mean only (no information)"
MAJ_NAME = "Majority class (no information)"
META_NAME = "Metadata only"
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

has_position = POSITION_COL in wide.columns
has_season = SEASON_COL in wide.columns
has_stage = STAGE_COL in wide.columns

CLASS_TARGETS = {}
if has_position:
    CLASS_TARGETS["Position (Side vs Under)"] = POSITION_COL
if has_stage:
    CLASS_TARGETS["Stage"] = STAGE_COL
if has_season:
    CLASS_TARGETS["Season"] = SEASON_COL

task_options = (["Number (ADH / days since death)"] if target_options else []) + \
               [f"Category: {k}" for k in CLASS_TARGETS]
if not task_options:
    st.error("No numeric outcome columns and no Position/Stage/Season columns found in the processed data.")
    st.stop()

group_options = [c for c in wide.columns if c not in feature_like and c != "sample.id"]
group_default = group_options.index("Carcass") if "Carcass" in group_options else 0

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
task = st.radio("What do you want to predict?", task_options, horizontal=True, key="ml_task")
is_cls = task.startswith("Category")
cls_col = CLASS_TARGETS[task.replace("Category: ", "")] if is_cls else None
target = None

c1, c2, c3 = st.columns(3)
with c1:
    if is_cls:
        min_class = st.number_input("Drop classes with fewer samples than", 2, 50, 5, key="ml_minclass",
                                    help="Very rare classes can't be learned or validated.")
        drop_pipe = False
        if cls_col == STAGE_COL:
            drop_pipe = st.checkbox("Exclude dual-track Stage labels (containing '|')", True, key="ml_droppipe",
                                    help="Summer 'X | Y' stages are two parallel tracks (stillborn vs adult), "
                                         "not a single stage.")
    else:
        target = st.selectbox("Predict", target_options, key="ml_target",
                              help="PMI is usually the days-since-death column; ADH is accumulated degree hours.")
with c2:
    group_col = st.selectbox("Keep together in CV (never split)", group_options, index=group_default,
                             key="ml_group", help="Should be the carcass ID.")
with c3:
    if has_position and cls_col != POSITION_COL:
        positions = sorted(wide[POSITION_COL].dropna().astype(str).unique().tolist())
        pos_scope = st.selectbox("Sampling position", ["All"] + positions, key="ml_pos")
    else:
        pos_scope = "All"
    if has_season and cls_col != SEASON_COL:
        seasons_avail = sorted(wide[SEASON_COL].dropna().astype(str).unique().tolist())
        season_scope = st.selectbox("Season (analyse within one season)", ["All"] + seasons_avail, key="ml_season",
                                    help="Restricts everything to carcasses from one season, so the models can't "
                                         "use season as a shortcut. Fewer carcasses, so scores are noisier.")
    else:
        season_scope = "All"

d1, d2, d3, d4 = st.columns(4)
with d1:
    cv_choice = st.radio("Validation", ["Leave one carcass out", "Grouped k-fold"], key="ml_cv")
    k_folds = st.number_input("k (grouped k-fold)", 2, 20, 5, key="ml_k", disabled=cv_choice.startswith("Leave"))
with d2:
    prevalence = st.slider("Keep taxa present in at least (% of samples)", 0, 50, 10, 5, key="ml_prev")
with d3:
    add_meta = st.checkbox("Also give taxa models Season + Position as features", False, key="ml_addmeta",
                           help="Whichever of Season/Position is the thing being predicted is never used.")
with d4:
    use_log = False
    if not is_cls:
        use_log = st.checkbox("Log-transform the outcome", False, key="ml_log",
                              help="Helps if the outcome is strongly right-skewed. Scores are always on the original scale.")

with st.expander("Model settings"):
    s1, s2, s3, s4 = st.columns(4)
    rf_trees = s1.number_input("Random Forest trees", 100, 2000, 500, 100, key="ml_rf")
    gbm_trees = s2.number_input("Gradient Boosting trees", 50, 1000, 300, 50, key="ml_gbm")
    ridge_alpha = s3.number_input("Ridge alpha (numbers)", 0.01, 10000.0, 100.0, key="ml_ridge")
    kernel_alpha = s4.number_input("Kernel ridge alpha (numbers)", 0.001, 100.0, 0.1, key="ml_kalpha")
    s5, s6, _, _ = st.columns(4)
    logit_c = s5.number_input("Logistic C (categories)", 0.001, 100.0, 0.05, key="ml_logitc",
                              help="Smaller = stronger regularisation.")
    svm_c = s6.number_input("SVM C (categories)", 0.01, 100.0, 1.0, key="ml_svmc")

# ---------------------------------------------------------------------------
# Subset
# ---------------------------------------------------------------------------
if is_cls:
    lab = wide[cls_col]
    mask = lab.notna() & wide[group_col].notna()
    if cls_col == STAGE_COL and drop_pipe:
        mask &= ~lab.astype(str).str.contains("|", regex=False)
    if pos_scope != "All":
        mask &= wide[POSITION_COL].astype(str) == pos_scope
    if season_scope != "All":
        mask &= wide[SEASON_COL].astype(str) == season_scope
    counts = lab[mask].astype(str).value_counts()
    mask &= lab.astype(str).isin(counts[counts >= min_class].index)
else:
    mask = wide[target].notna() & wide[group_col].notna()
    if pos_scope != "All":
        mask &= wide[POSITION_COL].astype(str) == pos_scope
    if season_scope != "All":
        mask &= wide[SEASON_COL].astype(str) == season_scope

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

if is_cls:
    cls_counts = sub[cls_col].astype(str).value_counts()
    st.caption("Class sizes: " + ", ".join(f"{k}: {v}" for k, v in cls_counts.items()))
    if len(cls_counts) < 2:
        st.error("Need at least 2 classes with enough samples. Lower the minimum class size or change filters.")
        st.stop()

if n_groups < 3:
    st.error("Need at least 3 carcasses to cross-validate by carcass"
             + (" -- this season has too few. Choose All seasons or another season." if season_scope != "All" else "."))
    st.stop()
if season_scope != "All":
    st.info(f"Within-season analysis ({season_scope}): only {n_groups} carcasses, so each fold holds out a large "
            "share of the data and scores will be noisy. Skip the final test set here unless you have 7+ carcasses, "
            "and compare against the baselines rather than reading absolute numbers.")
if rel.shape[1] < 2:
    st.error("Fewer than 2 taxa left after the prevalence filter -- lower the threshold.")
    st.stop()
if use_log and (sub[target] < 0).any():
    st.error("Outcome has negative values, so it can't be log-transformed.")
    st.stop()

t1, t2, t3 = st.columns(3)
with t1:
    use_test = st.checkbox("Reserve carcasses as a final test set", False, key="ml_usetest",
                           help="These carcasses are removed before any model comparison and scored once at the end.")
max_test = max(1, n_groups - 3)
with t2:
    n_test = st.number_input("Carcasses to reserve", 1, max_test, min(3, max_test), key="ml_ntest",
                             disabled=not use_test)
with t3:
    seed = st.number_input("Random seed for the split", 0, 9999, 42, key="ml_seed", disabled=not use_test)
if use_test and n_groups - int(n_test) < 3:
    st.error("Keep at least 3 carcasses for cross-validation: reserve fewer carcasses for the test set.")
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


def _meta_matrix(exclude=None):
    """Season/Position indicator columns, never including the column being predicted."""
    cols = [c for c in (SEASON_COL, POSITION_COL) if c in sub.columns and c != exclude]
    if not cols:
        return np.zeros((len(sub), 0)), []
    df = pd.get_dummies(sub[cols].fillna("NA").astype(str))
    return df.to_numpy(float), list(df.columns)


def _dev_test_split(groups: np.ndarray):
    if not use_test:
        return np.arange(len(groups)), np.array([], dtype=int), []
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(int(seed))
    held = sorted(rng.choice(uniq, size=int(n_test), replace=False).tolist())
    is_test = np.isin(groups, held)
    return np.where(~is_test)[0], np.where(is_test)[0], held


def _cv_splits(dev_idx: np.ndarray, groups: np.ndarray):
    g = groups[dev_idx]
    splitter = LeaveOneGroupOut() if cv_choice.startswith("Leave") else GroupKFold(
        n_splits=int(min(k_folds, len(set(g)))))
    return [(dev_idx[a], dev_idx[b]) for a, b in splitter.split(np.zeros(len(dev_idx)), groups=g)]


def _reg_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    ok = len(y_true) > 2 and np.std(y_pred) > 0 and np.std(y_true) > 0
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - np.sum(err ** 2) / sst) if sst > 0 else np.nan,
        "Pearson r": float(pearsonr(y_true, y_pred)[0]) if ok else np.nan,
        "Spearman": float(spearmanr(y_true, y_pred)[0]) if ok else np.nan,
    }


def _cls_scores(y_true: np.ndarray, y_pred: np.ndarray, proba, k: int) -> dict:
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Macro F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ROC AUC": np.nan,
    }
    if proba is not None and not np.isnan(proba).any() and len(np.unique(y_true)) == k:
        try:
            out["ROC AUC"] = float(
                roc_auc_score(y_true, proba[:, 1]) if k == 2
                else roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(range(k))))
        except ValueError:
            pass
    return out


def _proba_full(model, x, k: int) -> np.ndarray:
    full = np.zeros((len(x), k))
    full[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(x)
    return full


# ---------------------------------------------------------------------------
# Numeric outcome
# ---------------------------------------------------------------------------
def _run_reg() -> dict:
    y_raw = sub[target].to_numpy(float)
    y = np.log1p(y_raw) if use_log else y_raw
    groups = sub[group_col].astype(str).to_numpy()
    n = len(sub)
    dev_idx, test_idx, held = _dev_test_split(groups)

    x_rel = rel.to_numpy(float)
    x_clr = _clr(x_rel)
    x_meta, meta_names = _meta_matrix()
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
        models[META_NAME + " (Season + Position)"] = meta_fn
    taxa_models = ["Random Forest", "Gradient Boosting", "Ridge (CLR)", "Kernel ridge (Jensen-Shannon)"]
    models.update(dict(zip(taxa_models, [rf_fn, gbm_fn, ridge_fn, kernel_fn])))

    splits = _cv_splits(dev_idx, groups)
    oof = {name: np.full(n, np.nan) for name in models}
    bar = st.progress(0.0, text="Cross-validating...")
    for i, (tr, te) in enumerate(splits):
        for name, fn in models.items():
            oof[name][te] = fn(tr, te)
        bar.progress((i + 1) / len(splits), text=f"Fold {i + 1} of {len(splits)}")
    bar.empty()
    preds = pd.DataFrame(oof)
    preds[ENSEMBLE_NAME] = preds[taxa_models].mean(axis=1)

    test_preds = None
    if len(test_idx):
        tp = {name: fn(dev_idx, test_idx) for name, fn in models.items()}
        test_preds = pd.DataFrame(tp)
        test_preds[ENSEMBLE_NAME] = test_preds[taxa_models].mean(axis=1)

    if use_log:
        preds = np.expm1(preds)
        if test_preds is not None:
            test_preds = np.expm1(test_preds)

    preds_dev = preds.iloc[dev_idx].reset_index(drop=True)
    y_dev = y_raw[dev_idx]
    scores = pd.DataFrame([{"Model": nm, **_reg_scores(y_dev, preds_dev[nm].to_numpy())} for nm in preds_dev.columns])
    base_mae = float(scores.loc[scores["Model"] == MEAN_NAME, "MAE"].iloc[0])
    scores["MAE improvement vs mean-only"] = 1 - scores["MAE"] / base_mae

    test_scores = None
    if test_preds is not None:
        y_test = y_raw[test_idx]
        test_scores = pd.DataFrame([{"Model": nm, **_reg_scores(y_test, test_preds[nm].to_numpy())}
                                    for nm in test_preds.columns])
        tb = float(test_scores.loc[test_scores["Model"] == MEAN_NAME, "MAE"].iloc[0])
        test_scores["MAE improvement vs mean-only"] = 1 - test_scores["MAE"] / tb if tb > 0 else np.nan

    imp_model = RandomForestRegressor(n_estimators=int(rf_trees), min_samples_leaf=2, max_features="sqrt",
                                      random_state=0, n_jobs=-1).fit(x_rf[dev_idx], y[dev_idx])
    names = list(rel.columns) + (meta_names if add_meta else [])
    importance = (pd.Series(imp_model.feature_importances_, index=names)
                  .sort_values(ascending=False).head(15).rename("importance").reset_index()
                  .rename(columns={"index": "feature"}))

    keep = [c for c in dict.fromkeys([group_col, SEASON_COL, POSITION_COL]) if c in sub.columns]
    detail = sub.iloc[dev_idx][keep].reset_index(drop=True)
    detail.insert(0, "observed", y_dev)
    test_detail = None
    if test_preds is not None:
        test_detail = sub.iloc[test_idx][keep].reset_index(drop=True)
        test_detail.insert(0, "observed", y_raw[test_idx])
        for c in test_preds.columns:
            test_detail[f"pred_{c}"] = test_preds[c].to_numpy()
    return {
        "kind": "reg", "preds": preds_dev, "detail": detail, "scores": scores, "importance": importance,
        "test_scores": test_scores, "test_detail": test_detail, "held": held,
        "group_col": group_col, "target": target, "n_folds": len(splits),
        "settings": _settings_text(target, len(dev_idx), len(set(groups[dev_idx]))),
    }


# ---------------------------------------------------------------------------
# Category outcome
# ---------------------------------------------------------------------------
def _run_cls() -> dict:
    labels = sub[cls_col].astype(str).to_numpy()
    classes = sorted(set(labels))
    k = len(classes)
    y = np.array([classes.index(v) for v in labels])
    groups = sub[group_col].astype(str).to_numpy()
    n = len(sub)
    dev_idx, test_idx, held = _dev_test_split(groups)

    x_rel = rel.to_numpy(float)
    x_clr = _clr(x_rel)
    x_meta, meta_names = _meta_matrix(exclude=cls_col)
    x_rf = np.hstack([x_rel, x_meta]) if add_meta else x_rel
    x_lin = np.hstack([x_clr, x_meta]) if add_meta else x_clr
    kern = _jsd_kernel(x_rel)

    def maj_fn(tr, te):
        cnt = np.bincount(y[tr], minlength=k)
        return np.full(len(te), int(np.argmax(cnt))), np.tile(cnt / cnt.sum(), (len(te), 1))

    def prob_model(make, x):
        def fn(tr, te):
            if len(np.unique(y[tr])) < 2:
                return maj_fn(tr, te)
            m = make()
            m.fit(x[tr], y[tr])
            p = _proba_full(m, x[te], k)
            return p.argmax(axis=1), p
        return fn

    def svm_fn(tr, te):
        if len(np.unique(y[tr])) < 2:
            return maj_fn(tr, te)
        m = SVC(kernel="precomputed", C=float(svm_c), class_weight="balanced")
        m.fit(kern[np.ix_(tr, tr)], y[tr])
        return m.predict(kern[np.ix_(te, tr)]), None

    models = {MAJ_NAME: maj_fn}
    if x_meta.shape[1]:
        models[META_NAME] = prob_model(lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=0, n_jobs=-1), x_meta)
    models["Random Forest"] = prob_model(lambda: RandomForestClassifier(
        n_estimators=int(rf_trees), min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced_subsample", random_state=0, n_jobs=-1), x_rf)
    models["Gradient Boosting"] = prob_model(lambda: GradientBoostingClassifier(
        n_estimators=int(gbm_trees), max_depth=3, learning_rate=0.05, subsample=0.8,
        random_state=0), x_rf)
    models["Logistic (CLR)"] = prob_model(lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=float(logit_c), max_iter=3000, class_weight="balanced")), x_lin)
    models["SVM (Jensen-Shannon kernel)"] = svm_fn
    prob_names = ["Random Forest", "Gradient Boosting", "Logistic (CLR)"]

    splits = _cv_splits(dev_idx, groups)
    oof_lab = {nm: np.full(n, -1) for nm in models}
    oof_pro = {nm: np.full((n, k), np.nan) for nm in models}
    bar = st.progress(0.0, text="Cross-validating...")
    for i, (tr, te) in enumerate(splits):
        for nm, fn in models.items():
            lab_, pro_ = fn(tr, te)
            oof_lab[nm][te] = lab_
            if pro_ is not None:
                oof_pro[nm][te] = pro_
        bar.progress((i + 1) / len(splits), text=f"Fold {i + 1} of {len(splits)}")
    bar.empty()

    def add_ensemble(lab_d, pro_d):
        ens = np.mean([pro_d[nm] for nm in prob_names], axis=0)
        lab_d[ENSEMBLE_NAME] = np.where(np.isnan(ens).any(axis=1), -1, np.nan_to_num(ens).argmax(axis=1))
        pro_d[ENSEMBLE_NAME] = ens

    add_ensemble(oof_lab, oof_pro)
    names_all = list(oof_lab)

    y_dev = y[dev_idx]

    def table(idx, lab_d, pro_d):
        rows = [{"Model": nm, **_cls_scores(y[idx], lab_d[nm][idx], pro_d[nm][idx], k)} for nm in names_all]
        df = pd.DataFrame(rows)
        maj_acc = float(df.loc[df["Model"] == MAJ_NAME, "Accuracy"].iloc[0])
        df["Accuracy gain vs majority"] = df["Accuracy"] - maj_acc
        return df

    scores = table(dev_idx, oof_lab, oof_pro)

    test_scores = test_detail = None
    if len(test_idx):
        t_lab, t_pro = {}, {}
        for nm, fn in models.items():
            lab_, pro_ = fn(dev_idx, test_idx)
            full_l = np.full(n, -1)
            full_l[test_idx] = lab_
            full_p = np.full((n, k), np.nan)
            if pro_ is not None:
                full_p[test_idx] = pro_
            t_lab[nm], t_pro[nm] = full_l, full_p
        add_ensemble(t_lab, t_pro)
        test_scores = table(test_idx, t_lab, t_pro)
        keep = [c for c in dict.fromkeys([group_col, SEASON_COL, POSITION_COL]) if c in sub.columns]
        test_detail = sub.iloc[test_idx][keep].reset_index(drop=True)
        test_detail.insert(0, "observed", [classes[i] for i in y[test_idx]])
        for nm in names_all:
            test_detail[f"pred_{nm}"] = [classes[i] if i >= 0 else "" for i in t_lab[nm][test_idx]]

    imp_model = RandomForestClassifier(
        n_estimators=int(rf_trees), min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced_subsample", random_state=0, n_jobs=-1).fit(x_rf[dev_idx], y_dev)
    names = list(rel.columns) + (meta_names if add_meta else [])
    importance = (pd.Series(imp_model.feature_importances_, index=names)
                  .sort_values(ascending=False).head(15).rename("importance").reset_index()
                  .rename(columns={"index": "feature"}))

    keep = [c for c in dict.fromkeys([group_col, SEASON_COL, POSITION_COL]) if c in sub.columns]
    detail = sub.iloc[dev_idx][keep].reset_index(drop=True)
    detail.insert(0, "observed", [classes[i] for i in y_dev])
    pred_lab = pd.DataFrame({nm: oof_lab[nm][dev_idx] for nm in names_all})
    return {
        "kind": "cls", "classes": classes, "y_dev": y_dev, "pred_lab": pred_lab, "detail": detail,
        "scores": scores, "importance": importance, "test_scores": test_scores, "test_detail": test_detail,
        "held": held, "group_col": group_col, "target": cls_col if cls_col != POSITION_COL else "Position",
        "n_folds": len(splits),
        "settings": _settings_text(cls_col if cls_col != POSITION_COL else "Position",
                                   len(dev_idx), len(set(groups[dev_idx]))),
    }


def _settings_text(what, n_dev, g_dev) -> str:
    bits = [str(what), rank, f"position={pos_scope}", f"season={season_scope}", cv_choice,
            f"{n_dev} samples, {g_dev} carcasses in cross-validation", f"{rel.shape[1]} taxa"]
    if use_log:
        bits.append("log outcome")
    if add_meta:
        bits.append("+Season/Position features")
    if use_test:
        bits.append(f"{int(n_test)} carcasses reserved for final test (seed {int(seed)})")
    return " | ".join(bits)


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------
def _show_test(res: dict, fmt: dict):
    if res["test_scores"] is None:
        return
    st.subheader("Final test on reserved carcasses")
    st.caption(
        f"Reserved carcasses: {', '.join(res['held'])}. Each model was trained on all other carcasses and scored "
        "once on these. With only a few carcasses this is noisy: treat it as a sanity check on the "
        "cross-validated table, not a replacement for it, and don't change the seed to look for a nicer result."
    )
    st.dataframe(res["test_scores"].round(fmt), hide_index=True, use_container_width=True)
    with st.expander("Test-set predictions"):
        st.dataframe(res["test_detail"], hide_index=True, use_container_width=True)
        st.download_button("Download test predictions (CSV)", res["test_detail"].to_csv(index=False).encode("utf-8"),
                           file_name=f"test_predictions_{res['target']}.csv", mime="text/csv", key="ml_test_csv")


def _show_reg(res: dict):
    scores = res["scores"]
    fmt = {"MAE": 2, "RMSE": 2, "R2": 3, "Pearson r": 3, "Spearman": 3, "MAE improvement vs mean-only": 3}
    st.dataframe(scores.round(fmt), hide_index=True, use_container_width=True)
    st.caption(
        "MAE/RMSE are in the outcome's own units (lower is better). R2 below 0 means worse than always "
        "guessing the overall average. 'MAE improvement' is the fraction of the mean-only error removed."
    )
    base = [MEAN_NAME, META_NAME + " (Season + Position)"]
    taxa_scores = scores[~scores["Model"].isin(base)]
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
        fig2 = px.bar(per_group, x=res["group_col"], y="MAE", title=f"Error by held-out carcass ({model_choice})")
        fig2.update_layout(xaxis_title="Carcass", yaxis_title=f"MAE ({res['target']})")
        st.plotly_chart(fig2, use_container_width=True, key="ml_bycarcass")
        st.caption("A few carcasses with much larger error often point to a biologically different group "
                   "(e.g. the Summer stillborn vs adult tracks) rather than a modeling problem.")

    out = detail.copy()
    for c in pred_cols:
        out[f"pred_{c}"] = res["preds"][c].to_numpy()
    _show_test(res, fmt)
    return out


def _show_cls(res: dict):
    classes = res["classes"]
    k = len(classes)
    scores = res["scores"]
    fmt = {"Accuracy": 3, "Balanced accuracy": 3, "Macro F1": 3, "ROC AUC": 3, "Accuracy gain vs majority": 3}
    st.dataframe(scores.round(fmt), hide_index=True, use_container_width=True)
    st.caption(
        f"Chance level for balanced accuracy is {1 / k:.2f} ({k} classes). Accuracy alone can mislead when classes are "
        "unequal, so look at balanced accuracy and macro F1 (average over classes). ROC AUC: 0.5 = chance, 1 = perfect; "
        "it needs probabilities, so the SVM has none."
    )
    base = [MAJ_NAME, META_NAME]
    taxa_scores = scores[~scores["Model"].isin(base)]
    best = taxa_scores.sort_values("Balanced accuracy", ascending=False).iloc[0]["Model"] if len(taxa_scores) \
        else ENSEMBLE_NAME
    pred_cols = list(res["pred_lab"].columns)
    model_choice = st.selectbox("Show details for", pred_cols, index=pred_cols.index(best), key="ml_show_cls")

    y_true = res["y_dev"]
    y_pred = res["pred_lab"][model_choice].to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    text = [[f"{cm[i, j]}<br>{pct[i, j]:.0%}" for j in range(k)] for i in range(k)]

    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Heatmap(z=pct, x=classes, y=classes, text=text, texttemplate="%{text}",
                                   colorscale="Blues", zmin=0, zmax=1, colorbar=dict(title="Share of true class")))
        fig.update_layout(title=f"{res['target']}: confusion matrix ({model_choice})",
                          xaxis_title="Predicted", yaxis_title="Observed", yaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True, key="ml_confusion")
        add_figure_download_buttons(fig, filename=f"confusion_{res['target']}_{model_choice.split()[0].lower()}",
                                    key_prefix="ml_conf_dl")
        st.caption("Each row is the observed class; each cell shows the count and the share of that row. "
                   "A strong diagonal means classes are recovered correctly.")
    with right:
        d = res["detail"].copy()
        d["correct"] = (y_true == y_pred).astype(float)
        per_group = (d.groupby(res["group_col"])["correct"].mean().sort_values().rename("Accuracy").reset_index())
        fig2 = px.bar(per_group, x=res["group_col"], y="Accuracy", title=f"Accuracy by held-out carcass ({model_choice})")
        fig2.update_layout(xaxis_title="Carcass", yaxis_range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True, key="ml_bycarcass_cls")

    out = res["detail"].copy()
    for nm in pred_cols:
        out[f"pred_{nm}"] = [classes[i] if i >= 0 else "" for i in res["pred_lab"][nm].to_numpy()]
    _show_test(res, fmt)
    return out


# ---------------------------------------------------------------------------
# Run + results
# ---------------------------------------------------------------------------
if st.button("Run benchmark", type="primary", key="ml_run_btn"):
    st.session_state["ml_result"] = _run_cls() if is_cls else _run_reg()

res = st.session_state.get("ml_result")
if res is None:
    st.info("Choose settings above, then click **Run benchmark**.")
    st.stop()

st.subheader("Results (cross-validated on held-out carcasses)")
st.caption(f"Settings: {res['settings']}  ({res['n_folds']} folds)")

out = _show_cls(res) if res["kind"] == "cls" else _show_reg(res)

with st.expander("Top taxa by Random Forest importance (fit on the cross-validation carcasses, for interpretation only)"):
    fig3 = px.bar(res["importance"].iloc[::-1], x="importance", y="feature", orientation="h")
    st.plotly_chart(fig3, use_container_width=True, key="ml_importance")

st.download_button("Download held-out predictions (CSV)", out.to_csv(index=False).encode("utf-8"),
                   file_name=f"heldout_predictions_{res['target']}.csv", mime="text/csv", key="ml_csv")

with st.expander("How to read this / cautions"):
    st.markdown(
        "- **Stage is not used as a predictor for ADH or days since death**: it is defined from how decomposed "
        "the carcass looks, so it would leak the answer. When *Stage itself* is the target, Season and Position "
        "may be used as features, but never the column being predicted.\n"
        "- **Position (Side vs Under)** is a two-class problem; chance is 50% balanced accuracy. A high score means "
        "the soil community differs clearly between Side and Under.\n"
        "- **Stage is treated as unordered categories.** Mistakes between neighbouring stages (e.g. Active vs "
        "Advanced) are less serious than far-apart ones, which the confusion matrix lets you see.\n"
        "- **The final test set** is only a few carcasses, so its numbers are noisy. Use it once; don't re-draw the "
        "split until it looks good.\n"
        "- With ~17 carcasses the estimates are noisy: one odd carcass can move every score. Look at the "
        "per-carcass plot, not just the summary table.\n"
        "- Following Maltecca et al. 2019, averaging across model types is usually safer than picking a single winner."
    )