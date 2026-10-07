# src/modeling.py
"""
Predictive modeling utilities mirroring the forensic microbiome papers:

  - Random forest regression of microbial community composition against
    collection date / ADH (Kaszubinski et al. 2022 built separate models
    for collection date and ADD, reporting OOB R^2 and mean squared error;
    top predictors were taken from feature importance).
  - Quadratic regression of individual taxa (or taxon counts) across
    collection date/ADH, matching Figure 3 in the 2022 paper (number of
    taxa regressed against collection date with a 95% CI band) and the
    "successional trajectory" framing used for phyla like Firmicutes.

Both operate on the `combined_long` -> pivoted wide relative-abundance
table (samples x taxa) plus a target Series (e.g. ADH or a numeric
collection-date code) aligned by sample.id.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance


def pivot_relabund_wide(long_df: pd.DataFrame, rank: str, sample_id_col: str = "sample.id") -> pd.DataFrame:
    """
    Pivots the long-format combined_long table (as produced by
    wrangle.py's melt_to_long) back to a sample x taxon wide table of
    relative abundances for a single rank ("Family" or "Phylum"),
    suitable as the feature matrix for random forest modeling.
    """
    sub = long_df[long_df["rank"] == rank]
    wide = sub.pivot_table(index=sample_id_col, columns="taxon", values="rel_abund", aggfunc="mean").fillna(0.0)
    return wide


def random_forest_pmsi(
    features: pd.DataFrame,
    target: pd.Series,
    n_trees: int = 500,
    n_splits: int = 5,
    random_state: int = 0,
) -> dict:
    """
    Fits a random forest regressor of `target` (e.g. ADH or collection
    date) on `features` (sample x taxon relative abundance), evaluated
    with K-fold out-of-sample prediction so the reported R^2 is not
    inflated by in-sample fit -- analogous to the OOB error reported in
    the papers' randomForest (R) models.

    Returns a dict with the fitted model (trained on all data), the
    cross-validated predictions, R^2, mean squared error, and a feature
    importance table (mirrors "top predictors ... based on largest
    increased node purity" in the 2022 paper).
    """
    aligned_target = target.loc[features.index]
    mask = aligned_target.notna()
    X = features.loc[mask]
    y = aligned_target.loc[mask]

    n_splits = max(2, min(n_splits, len(y)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base_model = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
    cv_pred = cross_val_predict(base_model, X, y, cv=kf)

    ss_res = np.sum((y.to_numpy() - cv_pred) ** 2)
    ss_tot = np.sum((y.to_numpy() - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mse = np.mean((y.to_numpy() - cv_pred) ** 2)

    final_model = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
    final_model.fit(X, y)
    importances = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    predictions = pd.DataFrame({"true": y, "predicted": cv_pred}, index=y.index)

    return {
        "model": final_model,
        "predictions": predictions,
        "r2": r2,
        "mse": mse,
        "n_samples": len(y),
        "feature_importance": importances,
    }


def quadratic_trajectory(x: pd.Series, y: pd.Series, n_points: int = 100) -> dict:
    """
    Fits y ~ b0 + b1*x + b2*x^2 (matching the "Quadratic regression models"
    used for core/internal/external taxa counts and phylum trajectories
    across collection date/ADH in Kaszubinski et al. 2022, Figure 3),
    returning fitted curve points plus an approximate 95% confidence band
    computed from the regression's prediction standard error.
    """
    mask = x.notna() & y.notna()
    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=float)

    # A quadratic fit has 3 parameters (intercept, linear, quadratic);
    # estimating a covariance for the confidence band needs more data
    # points than that, or numpy's polyfit(..., cov=True) fails with a
    # broadcasting error (0 residual degrees of freedom). Fail gracefully
    # (return None) rather than crashing -- callers should check for this,
    # e.g. after filtering to a small subset with few samples.
    if len(xv) < 4:
        return None

    try:
        coeffs, cov = np.polyfit(xv, yv, deg=2, cov=True)
    except (ValueError, np.linalg.LinAlgError):
        return None
    x_grid = np.linspace(xv.min(), xv.max(), n_points)
    design = np.vstack([x_grid ** 2, x_grid, np.ones_like(x_grid)]).T
    y_fit = design @ coeffs

    # Prediction variance from the covariance of the fitted coefficients.
    pred_var = np.einsum("ij,jk,ik->i", design, cov, design)
    resid = yv - (np.vstack([xv ** 2, xv, np.ones_like(xv)]).T @ coeffs)
    dof = max(len(xv) - 3, 1)
    resid_var = np.sum(resid ** 2) / dof
    se = np.sqrt(np.clip(pred_var, 0, None) + resid_var)
    ci95 = 1.96 * se

    y_pred_obs = np.vstack([xv ** 2, xv, np.ones_like(xv)]).T @ coeffs
    ss_res = np.sum((yv - y_pred_obs) ** 2)
    ss_tot = np.sum((yv - yv.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "x_grid": x_grid,
        "y_fit": y_fit,
        "y_lower": y_fit - ci95,
        "y_upper": y_fit + ci95,
        "coeffs": coeffs,  # [quadratic, linear, intercept]
        "r2": r2,
        "n": len(xv),
    }


def random_forest_classify(
    features: pd.DataFrame,
    target: pd.Series,
    n_trees: int = 500,
    n_splits: int = 5,
    random_state: int = 0,
) -> dict:
    """
    Fits a random forest CLASSIFIER of a categorical `target` (e.g. Stage,
    or sampling Position) on `features` (sample x taxon relative
    abundance), evaluated with stratified K-fold cross-validation so the
    reported accuracy isn't inflated by in-sample fit. Mirrors
    random_forest_pmsi() but for categorical targets, matching the papers'
    practice of building separate models for categorical (e.g. collection
    date as a category) vs. continuous (ADD) targets.

    Returns a dict with the fitted model, cross-validated predictions,
    accuracy, balanced accuracy (average per-class recall -- a fairer
    summary than raw accuracy when classes are imbalanced, e.g. a Stage
    with 90 samples vs. one with 12), confusion matrix (+ its label
    order), a per-class precision/recall/F1 report, and feature
    importances. Uses class_weight="balanced" so the model doesn't
    implicitly prioritize the majority class(es) purely because they have
    more training examples.
    """
    aligned_target = target.loc[features.index]
    mask = aligned_target.notna()
    X = features.loc[mask]
    y = aligned_target.loc[mask].astype(str)

    class_counts = y.value_counts()
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        # Can't stratify a class with fewer than 2 members -- fall back to
        # plain (non-stratified) KFold rather than erroring out.
        n_splits_eff = max(2, min(n_splits, len(y)))
        kf = KFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
        print(f"random_forest_classify: class(es) with only 1 sample present -- "
              f"using non-stratified KFold instead of StratifiedKFold.")
    else:
        n_splits_eff = max(2, min(n_splits, min_class_count))
        kf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)

    base_model = RandomForestClassifier(n_estimators=n_trees, random_state=random_state, class_weight="balanced")
    cv_pred = cross_val_predict(base_model, X, y, cv=kf)

    labels = sorted(y.unique().tolist())
    accuracy = accuracy_score(y, cv_pred)
    balanced_acc = balanced_accuracy_score(y, cv_pred)
    cm = confusion_matrix(y, cv_pred, labels=labels)
    report = classification_report(y, cv_pred, labels=labels, output_dict=True, zero_division=0)

    final_model = RandomForestClassifier(n_estimators=n_trees, random_state=random_state, class_weight="balanced")
    final_model.fit(X, y)
    importances = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    predictions = pd.DataFrame({"true": y, "predicted": cv_pred}, index=y.index)

    return {
        "model": final_model,
        "predictions": predictions,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": cm,
        "labels": labels,
        "classification_report": report,
        "n_samples": len(y),
        "n_splits_used": n_splits_eff,
        "feature_importance": importances,
    }


def fit_random_forest(
    features: pd.DataFrame,
    target: pd.Series,
    n_trees: int = 500,
    n_splits: int = 5,
    random_state: int = 0,
) -> dict:
    """
    Single entry point for the modeling page: auto-detects whether `target`
    is numeric (e.g. ADH, DaysSinceDeath -> regression via
    random_forest_pmsi) or categorical (e.g. Stage, Position, Season ->
    classification via random_forest_classify), fits the appropriate
    model, and returns its result dict with an added "task" key
    ("regression" or "classification") so calling code can branch on it.
    """
    if pd.api.types.is_numeric_dtype(target):
        result = random_forest_pmsi(features, target, n_trees=n_trees, n_splits=n_splits, random_state=random_state)
        result["task"] = "regression"
    else:
        result = random_forest_classify(features, target, n_trees=n_trees, n_splits=n_splits, random_state=random_state)
        result["task"] = "classification"
    return result


# ---------------------------------------------------------------------------
# Neural network (MLP) models -- for soil-provenance / "which carcass did
# this sample come from" style questions, as suggested alongside random
# forest. Built to return the SAME dict shape as the random forest
# functions above (r2/mse/predictions for regression; accuracy/
# confusion_matrix/labels/classification_report for classification) so the
# dashboard's results display can render either model type identically.
# MLPs have no built-in feature_importances_ (that's a tree-specific
# concept), so permutation importance is used instead -- it works for any
# estimator: shuffle one feature at a time, remeasure performance, and the
# resulting performance DROP is that feature's importance. This is slower
# than a tree's built-in importance but is the standard model-agnostic
# substitute.
# ---------------------------------------------------------------------------


def mlp_regress(
    features: pd.DataFrame,
    target: pd.Series,
    hidden_layer_sizes: tuple = (64, 32),
    n_splits: int = 5,
    random_state: int = 0,
    max_iter: int = 2000,
) -> dict:
    """
    Fits a small feedforward neural network (MLPRegressor) to predict a
    numeric target, cross-validated the same way as random_forest_pmsi so
    the two model types are directly comparable. Features are standardized
    first (neural nets are sensitive to input scale in a way tree models
    aren't).
    """
    aligned_target = target.loc[features.index]
    mask = aligned_target.notna()
    X = features.loc[mask]
    y = aligned_target.loc[mask]

    n_splits_eff = max(2, min(n_splits, len(y)))
    kf = KFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
    base_model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state, max_iter=max_iter),
    )
    cv_pred = cross_val_predict(base_model, X, y, cv=kf)

    ss_res = np.sum((y.to_numpy() - cv_pred) ** 2)
    ss_tot = np.sum((y.to_numpy() - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mse = np.mean((y.to_numpy() - cv_pred) ** 2)

    final_model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state, max_iter=max_iter),
    )
    final_model.fit(X, y)
    perm = permutation_importance(final_model, X, y, n_repeats=10, random_state=random_state)
    importances = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

    predictions = pd.DataFrame({"true": y, "predicted": cv_pred}, index=y.index)

    return {
        "model": final_model,
        "predictions": predictions,
        "r2": r2,
        "mse": mse,
        "n_samples": len(y),
        "feature_importance": importances,
    }


def mlp_classify(
    features: pd.DataFrame,
    target: pd.Series,
    hidden_layer_sizes: tuple = (64, 32),
    n_splits: int = 5,
    random_state: int = 0,
    max_iter: int = 2000,
) -> dict:
    """
    Fits a small feedforward neural network (MLPClassifier) to predict a
    categorical target (e.g. Stage, or -- the soil-provenance use case --
    Carcass, to test whether microbial composition can identify which
    specific carcass/site a sample came from). Mirrors
    random_forest_classify's evaluation and fallback behavior (including
    falling back to non-stratified KFold when a class has fewer than 2
    members, which is common with many classes and a small sample size,
    e.g. many distinct Carcass IDs).
    """
    aligned_target = target.loc[features.index]
    mask = aligned_target.notna()
    X = features.loc[mask]
    y = aligned_target.loc[mask].astype(str)

    class_counts = y.value_counts()
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        n_splits_eff = max(2, min(n_splits, len(y)))
        kf = KFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
        print(f"mlp_classify: class(es) with only 1 sample present -- "
              f"using non-stratified KFold instead of StratifiedKFold.")
    else:
        n_splits_eff = max(2, min(n_splits, min_class_count))
        kf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)

    base_model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state, max_iter=max_iter),
    )
    cv_pred = cross_val_predict(base_model, X, y, cv=kf)

    labels = sorted(y.unique().tolist())
    accuracy = accuracy_score(y, cv_pred)
    cm = confusion_matrix(y, cv_pred, labels=labels)
    report = classification_report(y, cv_pred, labels=labels, output_dict=True, zero_division=0)

    final_model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state, max_iter=max_iter),
    )
    final_model.fit(X, y)
    perm = permutation_importance(final_model, X, y, n_repeats=10, random_state=random_state)
    importances = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

    predictions = pd.DataFrame({"true": y, "predicted": cv_pred}, index=y.index)

    return {
        "model": final_model,
        "predictions": predictions,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "labels": labels,
        "classification_report": report,
        "n_samples": len(y),
        "n_splits_used": n_splits_eff,
        "feature_importance": importances,
    }


def fit_neural_net(
    features: pd.DataFrame,
    target: pd.Series,
    hidden_layer_sizes: tuple = (64, 32),
    n_splits: int = 5,
    random_state: int = 0,
) -> dict:
    """
    Neural network counterpart to fit_random_forest() -- same auto-dispatch
    on target dtype (numeric -> mlp_regress, categorical -> mlp_classify),
    same "task" key added to the result, so the dashboard can call either
    fit_random_forest() or fit_neural_net() interchangeably and render the
    result with identical display code.
    """
    if pd.api.types.is_numeric_dtype(target):
        result = mlp_regress(features, target, hidden_layer_sizes=hidden_layer_sizes,
                              n_splits=n_splits, random_state=random_state)
        result["task"] = "regression"
    else:
        result = mlp_classify(features, target, hidden_layer_sizes=hidden_layer_sizes,
                               n_splits=n_splits, random_state=random_state)
        result["task"] = "classification"
    return result