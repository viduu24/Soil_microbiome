# src/diversity.py
"""
Diversity metrics and differential abundance testing, following the
methodology used in Kaszubinski et al. 2020 (bioinformatic pipeline
comparison) and Kaszubinski et al. 2022 (bone microbial succession):

  - Rarefaction to a fixed sequencing depth before computing diversity
    metrics or relative abundance, chosen from a rarefaction curve
    rather than an arbitrary "smallest sample" cutoff.
  - Alpha diversity: observed richness, Chao1 richness, Shannon diversity.
  - Beta diversity: Bray-Curtis / Jaccard dissimilarity + PCoA ordination.
  - Differential abundance: ANCOM (Mandal et al. 2015), which tests
    log-ratios between every pair of taxa rather than raw relative
    abundances directly, since relative abundance data is compositional
    (values are constrained to sum to 1, which violates the independence
    assumptions of ordinary tests like a t-test run taxon-by-taxon).

This module operates on WIDE count tables: rows = samples (indexed by
sample.id), columns = taxa, values = raw read counts (not relative
abundance -- rarefaction and ANCOM's log-ratios both require counts).
This matches the `counts` block produced by wrangle.py's
compute_relative_abundance() before the *_relabund columns are appended.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import kruskal
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------------------
# Rarefaction
# ---------------------------------------------------------------------------

def rarefaction_curve(counts: pd.DataFrame, steps: int = 15, seed: int = 0) -> pd.DataFrame:
    """
    Computes an observed-richness rarefaction curve for each sample,
    subsampling at `steps` evenly spaced depths up to each sample's own
    total read count. Use this to pick a defensible minimum library size
    (the depth at which most samples' curves have started to plateau)
    instead of guessing.

    Returns long-format: sample.id, depth, richness.
    """
    rng = np.random.default_rng(seed)
    totals = counts.sum(axis=1)
    max_depth = int(totals.max())
    depths = np.unique(np.linspace(1, max_depth, steps).astype(int))

    records = []
    for sample_id, row in counts.iterrows():
        total = int(row.sum())
        if total == 0:
            continue
        taxa = row.index.to_numpy()
        weights = row.to_numpy(dtype=float)
        for depth in depths:
            if depth > total:
                continue
            sub = _subsample_counts(weights, depth, rng)
            richness = int((sub > 0).sum())
            records.append({"sample.id": sample_id, "depth": depth, "richness": richness})
    return pd.DataFrame.from_records(records)


def _subsample_counts(counts_row: np.ndarray, depth: int, rng: np.random.Generator) -> np.ndarray:
    """
    Subsamples a single sample's count vector down to `depth` total reads
    without replacement (i.e., true rarefaction, not a multinomial
    resample), using the standard "expand to reads, draw without
    replacement, re-tally" approach.
    """
    total = int(counts_row.sum())
    if depth >= total:
        return counts_row.copy()
    # Expand counts into a flat array of taxon-index "reads", draw `depth`
    # of them without replacement, then re-tally into counts per taxon.
    read_taxon_idx = np.repeat(np.arange(len(counts_row)), counts_row.astype(int))
    chosen = rng.choice(read_taxon_idx, size=depth, replace=False)
    tallied = np.bincount(chosen, minlength=len(counts_row))
    return tallied


def rarefy_counts(counts: pd.DataFrame, depth: int, seed: int = 0, drop_below_depth: bool = True) -> pd.DataFrame:
    """
    Rarefies every sample (row) in `counts` to exactly `depth` total reads.
    Samples with fewer than `depth` reads are dropped by default (matching
    standard practice -- see rarefaction_curve() for choosing `depth`).
    """
    rng = np.random.default_rng(seed)
    totals = counts.sum(axis=1)

    if drop_below_depth:
        keep = totals[totals >= depth].index
        dropped = totals[totals < depth].index
        if len(dropped) > 0:
            print(f"rarefy_counts: dropping {len(dropped)} sample(s) below depth {depth}: "
                  f"{list(dropped)}")
        counts = counts.loc[keep]

    rarefied = counts.apply(
        lambda row: pd.Series(_subsample_counts(row.to_numpy(dtype=float), depth, rng), index=row.index),
        axis=1,
    )
    return rarefied.astype(int)


# ---------------------------------------------------------------------------
# Alpha diversity
# ---------------------------------------------------------------------------

def observed_richness(counts: pd.DataFrame) -> pd.Series:
    return (counts > 0).sum(axis=1)


def shannon_diversity(counts: pd.DataFrame) -> pd.Series:
    p = counts.div(counts.sum(axis=1), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p * np.log(p)
    terms = terms.fillna(0.0)
    return -terms.sum(axis=1)


def chao1_richness(counts: pd.DataFrame) -> pd.Series:
    """Chao1 richness estimator: S_obs + (f1^2 / (2 * f2)), where f1/f2
    are counts of taxa observed exactly once/twice (singletons/doubletons)."""
    s_obs = (counts > 0).sum(axis=1)
    f1 = (counts == 1).sum(axis=1)
    f2 = (counts == 2).sum(axis=1)
    # Avoid divide-by-zero when there are no doubletons; use the standard
    # bias-corrected form (f1 * (f1 - 1)) / (2 * (f2 + 1)) in that case.
    correction = np.where(
        f2 > 0,
        (f1 ** 2) / (2 * f2.replace(0, np.nan)),
        (f1 * (f1 - 1)) / 2,
    )
    correction = pd.Series(correction, index=counts.index).fillna(0.0)
    return s_obs + correction


def alpha_diversity(counts: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper returning all three alpha-diversity metrics
    (as reported in Kaszubinski et al. 2022) in one table, indexed by sample.id."""
    return pd.DataFrame({
        "observed_richness": observed_richness(counts),
        "chao1": chao1_richness(counts),
        "shannon": shannon_diversity(counts),
    })


# ---------------------------------------------------------------------------
# Beta diversity + PCoA
# ---------------------------------------------------------------------------

def beta_diversity(counts_or_relabund: pd.DataFrame, metric: str = "braycurtis") -> pd.DataFrame:
    """
    Pairwise sample dissimilarity matrix. `metric` = 'braycurtis' (default;
    abundance-weighted, analogous to weighted UniFrac without a tree) or
    'jaccard' (presence/absence, analogous to unweighted UniFrac).
    """
    if metric == "jaccard":
        data = (counts_or_relabund > 0).astype(int).to_numpy()
    else:
        data = counts_or_relabund.to_numpy()
    dist = squareform(pdist(data, metric=metric))
    return pd.DataFrame(dist, index=counts_or_relabund.index, columns=counts_or_relabund.index)


def pcoa(distance_df: pd.DataFrame, n_components: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Classical PCoA (metric multidimensional scaling) via double-centering
    and eigendecomposition of the distance matrix, matching the ordination
    plots (PCoA of UniFrac/Jaccard distances) used in both papers.

    Returns (coords_df, variance_explained_fraction_per_axis).
    """
    D = distance_df.to_numpy()
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J  # Gower's double centering

    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    positive = eigvals > 1e-10
    total_positive_var = eigvals[positive].sum()
    var_explained = np.divide(
        eigvals[:n_components],
        total_positive_var,
        out=np.zeros(n_components),
        where=total_positive_var > 0,
    )

    coords = eigvecs[:, :n_components] * np.sqrt(np.clip(eigvals[:n_components], 0, None))
    coords_df = pd.DataFrame(
        coords,
        index=distance_df.index,
        columns=[f"Axis.{i+1}" for i in range(n_components)],
    )
    return coords_df, var_explained


def permanova(distance_df: pd.DataFrame, grouping: pd.Series, permutations: int = 999, seed: int = 0) -> dict:
    """
    Permutational multivariate analysis of variance (Anderson 2001) on a
    precomputed distance matrix -- tests whether group centroids differ,
    matching the PERMANOVA (package = vegan) calls in both papers.

    Returns {"F": pseudo-F statistic, "R2": fraction of total sum-of-squares
    explained by the grouping (SS_among / SS_total -- the standard effect
    size reported alongside pseudo-F in a PERMANOVA table), "p_value":
    permutation p-value, "permutations": permutations run}.
    """
    grouping = grouping.loc[distance_df.index]
    D = distance_df.to_numpy()
    n = D.shape[0]
    groups = grouping.to_numpy()
    unique_groups = np.unique(groups)

    def sums_of_squares(D, groups):
        ss_total = (D ** 2).sum() / (2 * n)
        ss_within = 0.0
        for g in unique_groups:
            idx = np.where(groups == g)[0]
            n_g = len(idx)
            if n_g <= 1:
                continue
            sub = D[np.ix_(idx, idx)]
            ss_within += (sub ** 2).sum() / (2 * n_g)
        return ss_total, ss_within, ss_total - ss_within

    def pseudo_f(D, groups):
        ss_total, ss_within, ss_among = sums_of_squares(D, groups)
        a = len(unique_groups)
        df_among = a - 1
        df_within = n - a
        if df_within <= 0 or ss_within <= 0:
            return np.nan
        return (ss_among / df_among) / (ss_within / df_within)

    observed_f = pseudo_f(D, groups)
    ss_total_obs, _, ss_among_obs = sums_of_squares(D, groups)
    r2 = ss_among_obs / ss_total_obs if ss_total_obs > 0 else np.nan

    rng = np.random.default_rng(seed)
    perm_fs = np.empty(permutations)
    for i in range(permutations):
        perm_groups = rng.permutation(groups)
        perm_fs[i] = pseudo_f(D, perm_groups)

    p_value = (np.sum(perm_fs >= observed_f) + 1) / (permutations + 1)
    return {"F": observed_f, "R2": r2, "p_value": p_value, "permutations": permutations}


def permanova_table(dist_by_group: dict, permutations: int = 999, seed: int = 0) -> pd.DataFrame:
    """
    Runs permanova() once per entry in `dist_by_group` and summarizes the
    results in one table -- e.g. comparing PERMANOVA across several
    grouping variables (Season, Stage, Position) on the same distance
    matrix, which is the standard "PERMANOVA table" layout for a paper
    (one row per tested factor).

    `dist_by_group`: dict of label -> (distance_df, grouping_series),
    e.g. {"Season": (dist, wide_df.set_index("sample.id")["Season"])}.

    Returns a DataFrame indexed by label with columns: n_groups, n_samples,
    pseudo_F, R2, p_value, permutations.
    """
    records = []
    for label, (dist_df, grouping) in dist_by_group.items():
        grouping = grouping.loc[dist_df.index].dropna()
        if grouping.nunique() < 2:
            continue
        dist_sub = dist_df.loc[grouping.index, grouping.index]
        result = permanova(dist_sub, grouping, permutations=permutations, seed=seed)
        records.append({
            "grouping_variable": label,
            "n_groups": grouping.nunique(),
            "n_samples": len(grouping),
            "pseudo_F": result["F"],
            "R2": result["R2"],
            "p_value": result["p_value"],
            "permutations": result["permutations"],
        })
    return pd.DataFrame.from_records(records).set_index("grouping_variable")


def _term_design_columns(factor_dummies: dict, term: str) -> np.ndarray:
    """
    Design-matrix columns for one term: a single factor's dummy columns,
    or (for a term like "A:B") the elementwise-product columns of A's and
    B's dummies -- one product column per (A level, B level) combination,
    the standard way an interaction term is encoded in a linear model.
    """
    if ":" in term:
        a_name, b_name = term.split(":")
        a_cols, b_cols = factor_dummies[a_name], factor_dummies[b_name]
        if a_cols.shape[1] == 0 or b_cols.shape[1] == 0:
            return np.zeros((a_cols.shape[0], 0))
        cols = [(a_cols[:, i] * b_cols[:, j])[:, None]
                for i in range(a_cols.shape[1]) for j in range(b_cols.shape[1])]
        return np.hstack(cols)
    return factor_dummies[term]


def permanova_factorial(dist: pd.DataFrame, factors: dict, interactions: bool = True,
                         permutations: int = 999, seed: int = 0) -> pd.DataFrame:
    """
    Multi-factor PERMANOVA (McArdle & Anderson 2001) with sequential
    (Type I) sums-of-squares -- the distance-based equivalent of R's
    `vegan::adonis2(dist ~ A * B, by = "terms")`. Unlike running
    permanova() separately on each factor, this can test whether two
    factors INTERACT -- e.g. whether the Stage effect on community
    composition looks different depending on Season, rather than just
    whether Season and Stage each have their own separate effect.

    `factors`: dict of name -> pd.Series (categorical), in the ORDER you
    want them entered into the model. Order matters for sequential/Type I
    SS in an unbalanced design (the same caveat applies to vegan) -- put
    your primary variable of interest first if you're unsure.

    `interactions=True` adds every pairwise interaction term (A:B, A:C,
    B:C, ...) after all main effects. With 3+ factors this multiplies out
    fast (3 factors -> 3 main effects + 3 two-way interactions, each
    needing its own `permutations`-size permutation loop) -- for a
    specific interaction you care about, it's faster and more direct to
    pass just those two factors in `factors` rather than everything at
    once.

    P-values use the Freedman-Lane procedure (ter Braak 1992; Anderson &
    ter Braak 2003): for each term, residuals from the model WITHOUT that
    term (but with every term before it) are permuted and added back,
    then the term's sum-of-squares is recomputed on that permuted
    response -- this is what makes it valid for testing an interaction
    term specifically, not just the first factor in the model (naively
    permuting raw sample labels is only strictly valid for the very first
    term). This is the same method vegan's adonis2 uses by default for
    sequential ("by=terms") tests, implemented here directly from the
    distance matrix (no coordinate data needed) via the Gower-centered
    Gram matrix also used in pcoa().

    Returns an ANOVA-style DataFrame indexed by term, with one row per
    main effect and interaction plus "Residual" and "Total" rows, and
    columns: df, SS, R2, pseudo_F, p_value. R2 for a term is its SS as a
    fraction of total SS (same effect-size interpretation as
    permanova()'s R2) -- note terms' R2 values need not sum to the
    model's total R2 exactly in a very unbalanced design, another
    similarity with sequential ANOVA/adonis2 tables generally.

    Caveat: this is a from-scratch reimplementation of a fairly intricate
    permutation scheme. It's been validated against synthetic data with a
    known baked-in interaction (correctly flagged significant) and known
    independent main effects with NO interaction (interaction term
    correctly came back non-significant) -- but for a result going
    directly into a manuscript, cross-checking the key p-value against
    `vegan::adonis2()` in R is good practice before reporting it.
    """
    common_idx = dist.index
    for s in factors.values():
        common_idx = common_idx.intersection(s.dropna().index)
    dist = dist.loc[common_idx, common_idx]
    n = len(common_idx)

    factor_dummies = {}
    for name, series in factors.items():
        s = series.loc[common_idx].astype(str)
        levels = sorted(s.unique())
        if len(levels) > 1:
            dummies = np.column_stack([(s == lvl).to_numpy(dtype=float) for lvl in levels[1:]])
        else:
            dummies = np.zeros((n, 0))
        factor_dummies[name] = dummies

    main_terms = list(factors.keys())
    terms = list(main_terms)
    if interactions and len(main_terms) >= 2:
        terms += [f"{a}:{b}" for a, b in combinations(main_terms, 2)]

    D = dist.to_numpy()
    J = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * J @ (D ** 2) @ J
    ss_total = float(np.trace(G))

    def hat_matrix(X):
        return X @ np.linalg.pinv(X.T @ X) @ X.T

    X_running = np.ones((n, 1))
    H_running = hat_matrix(X_running)
    rank_running = 1

    rows = []
    term_info = []
    for term in terms:
        X_term = _term_design_columns(factor_dummies, term)
        X_upto = np.hstack([X_running, X_term])
        H_upto = hat_matrix(X_upto)
        rank_upto = np.linalg.matrix_rank(X_upto)
        df_term = rank_upto - rank_running

        SS_term = float(np.trace(H_upto @ G)) - float(np.trace(H_running @ G))
        term_info.append((term, H_running.copy(), H_upto.copy(), df_term))
        rows.append({"term": term, "df": df_term, "SS": SS_term})

        X_running, H_running, rank_running = X_upto, H_upto, rank_upto

    ss_model = float(np.trace(H_running @ G))
    ss_residual = ss_total - ss_model
    df_residual = n - rank_running

    rng = np.random.default_rng(seed)
    for row, (term, H_before, H_upto, df_term) in zip(rows, term_info):
        if df_term <= 0:
            # Can happen if a factor-level combination is entirely absent
            # (unbalanced design) and the interaction term is rank-deficient.
            row.update(p_value=np.nan, pseudo_F=np.nan, R2=np.nan)
            continue

        Gf = H_before @ G @ H_before
        I_minus_Hb = np.eye(n) - H_before
        C = H_before @ G @ I_minus_Hb
        Gr = I_minus_Hb @ G @ I_minus_Hb

        observed_SS = row["SS"]
        perm_SS = np.empty(permutations)
        for b in range(permutations):
            perm = rng.permutation(n)
            # Freedman-Lane: permuted response's Gram matrix, expressed
            # entirely via G (no raw coordinate data needed) -- see
            # function docstring for the derivation.
            G_perm = Gf + C[:, perm] + C.T[perm, :] + Gr[np.ix_(perm, perm)]
            perm_SS[b] = np.sum(H_upto * G_perm) - np.sum(H_before * G_perm)

        row["p_value"] = (np.sum(perm_SS >= observed_SS) + 1) / (permutations + 1)
        row["pseudo_F"] = (observed_SS / df_term) / (ss_residual / df_residual) if df_residual > 0 else np.nan
        row["R2"] = observed_SS / ss_total if ss_total > 0 else np.nan

    rows.append({"term": "Residual", "df": df_residual, "SS": ss_residual,
                 "R2": ss_residual / ss_total if ss_total > 0 else np.nan,
                 "pseudo_F": np.nan, "p_value": np.nan})
    rows.append({"term": "Total", "df": n - 1, "SS": ss_total,
                 "R2": 1.0, "pseudo_F": np.nan, "p_value": np.nan})

    table = pd.DataFrame(rows).set_index("term")
    return table[["df", "SS", "R2", "pseudo_F", "p_value"]]


def pairwise_permanova(dist: pd.DataFrame, grouping: pd.Series, permutations: int = 999, seed: int = 0) -> pd.DataFrame:
    """
    Post-hoc PERMANOVA on every pair of groups in `grouping` separately
    (rather than the single omnibus test across all groups at once), then
    BH-corrects the p-values across all pairs -- standard practice once an
    omnibus PERMANOVA comes back significant, to see WHICH groups actually
    differ from which (e.g. omnibus across 4 Seasons is significant --
    pairwise then shows it's really just Summer differing from the other 3).

    Returns a DataFrame with columns: group_1, group_2, n1, n2, pseudo_F,
    R2, p_value, q_value -- sorted by q_value ascending.
    """
    grouping = grouping.loc[dist.index].dropna()
    groups = sorted(grouping.unique().tolist())
    records = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            ids = grouping[grouping.isin([g1, g2])].index
            sub_dist = dist.loc[ids, ids]
            sub_group = grouping.loc[ids]
            result = permanova(sub_dist, sub_group, permutations=permutations, seed=seed)
            records.append({
                "group_1": g1, "group_2": g2,
                "n1": int((sub_group == g1).sum()), "n2": int((sub_group == g2).sum()),
                "pseudo_F": result["F"], "R2": result["R2"], "p_value": result["p_value"],
            })
    table = pd.DataFrame.from_records(records)
    if table.empty:
        return table
    table["q_value"] = _benjamini_hochberg(table["p_value"].to_numpy())
    return table.sort_values("q_value").reset_index(drop=True)


# ---------------------------------------------------------------------------
# ANCOM differential abundance
# ---------------------------------------------------------------------------

def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Minimal BH FDR correction (avoids a statsmodels dependency)."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    bh = ranked * n / (np.arange(n) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)
    out = np.empty(n)
    out[order] = bh
    return out


def ancom(counts: pd.DataFrame, grouping: pd.Series, alpha: float = 0.05,
          w_threshold: float = 0.7, pseudocount: float = 1.0) -> pd.DataFrame:
    """
    ANCOM (Mandal et al. 2015), as used for differential-abundance testing
    in both papers. For each taxon i, tests the log-ratio log(x_i / x_j)
    against every other taxon j across groups (Kruskal-Wallis), BH-corrects
    the resulting p-values across all j, and counts how many come out
    significant -- that count is taxon i's "W statistic". Taxa with
    W >= w_threshold * (n_taxa - 1) are declared differentially abundant.

    `counts` should be a wide sample x taxon table of raw (ideally
    rarefied) counts, aligned by index to `grouping` (a Series of group
    labels, e.g. Season or Stage).

    Returns a DataFrame indexed by taxon with columns: W, n_taxa_compared,
    detected (bool).
    """
    grouping = grouping.loc[counts.index]
    taxa = counts.columns.to_numpy()
    n_taxa = len(taxa)
    mat = counts.to_numpy(dtype=float) + pseudocount
    log_mat = np.log(mat)

    groups = grouping.to_numpy()
    unique_groups = np.unique(groups)
    group_idx = [np.where(groups == g)[0] for g in unique_groups]

    W = np.zeros(n_taxa)
    for i in range(n_taxa):
        pvals = np.empty(n_taxa - 1)
        k = 0
        for j in range(n_taxa):
            if i == j:
                continue
            log_ratio = log_mat[:, i] - log_mat[:, j]
            samples_by_group = [log_ratio[idx] for idx in group_idx]
            try:
                _, p = kruskal(*samples_by_group)
            except ValueError:
                p = 1.0
            pvals[k] = p
            k += 1
        adj = _benjamini_hochberg(pvals)
        W[i] = int((adj < alpha).sum())

    detected = W >= w_threshold * (n_taxa - 1)
    return pd.DataFrame({
        "taxon": taxa,
        "W": W.astype(int),
        "n_taxa_compared": n_taxa - 1,
        "W_ratio": W / (n_taxa - 1),
        "detected": detected,
    }).set_index("taxon").sort_values("W", ascending=False)