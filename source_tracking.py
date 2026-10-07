# src/source_tracking.py
"""
Simplified source tracking, in the spirit of FEAST (Shenhav et al. 2019)
as used in Kaszubinski et al. 2022 to estimate what fraction of an
internal bone community's taxa can be explained by the external bone
community (or by a prior timepoint), versus an "Unknown" source.

This is NOT the full FEAST expectation-maximization implementation (that
ships as an R package with a Gibbs-sampling / EM core over ASV-level
count data); it's a from-scratch EM point-estimate of source mixing
proportions using a multinomial mixture model, which recovers the same
quantity FEAST reports (percent contribution per source, including an
"Unknown" bucket for sink taxa the given sources don't explain well) at
the Family/Phylum rank data available here. Treat outputs as
directionally informative, not as a drop-in replacement for the R FEAST
package if you need publication-grade source tracking.
"""

import numpy as np
import pandas as pd


def estimate_source_contributions(
    sink_counts: pd.Series,
    source_counts: dict,
    unknown_prior_weight: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> pd.Series:
    """
    Estimates the proportion of a single sink sample's reads attributable
    to each source community (plus an "Unknown" source not explained by
    any provided source), via EM on a multinomial mixture:

        P(taxon | sink) = sum_k pi_k * P(taxon | source_k)

    `sink_counts` / each value in `source_counts`: Series indexed by
    taxon name, raw (or rarefied) read counts.
    `unknown_prior_weight`: Laplace-smoothing-style pseudo-uniform
    distribution standing in for "Unknown" -- taxa the sink has that no
    source explains well get attributed here, mirroring FEAST's unknown
    source category.

    Returns a Series of estimated proportions (sums to 1), indexed by
    source name (with "Unknown" as one of the entries).
    """
    taxa = sink_counts.index
    n_taxa = len(taxa)
    sink = sink_counts.reindex(taxa).fillna(0).to_numpy(dtype=float)
    sink_total = sink.sum()
    if sink_total == 0:
        raise ValueError("Sink sample has zero total reads.")

    source_names = list(source_counts.keys())
    source_profiles = []
    for name in source_names:
        s = source_counts[name].reindex(taxa).fillna(0).to_numpy(dtype=float)
        s = s + 1e-6  # avoid zero-probability taxa
        source_profiles.append(s / s.sum())
    # "Unknown" source: uniform-ish prior scaled by how idiosyncratic the
    # sink's own composition is, so it can absorb sink-specific taxa.
    unknown_profile = np.full(n_taxa, unknown_prior_weight) / (unknown_prior_weight * n_taxa)
    all_names = source_names + ["Unknown"]
    profiles = np.vstack(source_profiles + [unknown_profile])  # (n_sources+1, n_taxa)

    n_sources = profiles.shape[0]
    pi = np.full(n_sources, 1.0 / n_sources)

    for _ in range(max_iter):
        # E-step: expected reads-per-taxon attributable to each source.
        weighted = pi[:, None] * profiles  # (n_sources, n_taxa)
        denom = weighted.sum(axis=0)
        denom[denom == 0] = 1e-12
        responsibility = weighted / denom  # (n_sources, n_taxa)
        expected_counts = responsibility * sink[None, :]  # (n_sources, n_taxa)

        # M-step
        new_pi = expected_counts.sum(axis=1)
        new_pi = new_pi / new_pi.sum()

        if np.max(np.abs(new_pi - pi)) < tol:
            pi = new_pi
            break
        pi = new_pi

    return pd.Series(pi, index=all_names)


def estimate_contributions_over_time(
    sink_wide: pd.DataFrame,
    source_wide: pd.DataFrame,
    sink_meta: pd.DataFrame,
    source_meta: pd.DataFrame,
    time_col: str = "ADH",
    carcass_col: str = "Carcass",
) -> pd.DataFrame:
    """
    Convenience wrapper matching the 2022 paper's Figure 4: for each
    (carcass, timepoint) sink sample, estimates the percent contribution
    from that same carcass's source-community sample (e.g. external ->
    internal) at the same timepoint, plus Unknown.

    `sink_wide` / `source_wide`: sample.id x taxon count tables.
    `sink_meta` / `source_meta`: metadata with sample.id, carcass_col,
    and time_col, used to match sink samples to their corresponding
    source sample per carcass/timepoint.

    Returns long-format: sample.id, Carcass, ADH (or time_col), source,
    proportion.
    """
    records = []
    for _, sink_row_meta in sink_meta.iterrows():
        sink_id = sink_row_meta["sample.id"]
        if sink_id not in sink_wide.index:
            continue
        match = source_meta[
            (source_meta[carcass_col] == sink_row_meta[carcass_col])
            & (source_meta[time_col] == sink_row_meta[time_col])
        ]
        if match.empty:
            continue
        source_id = match.iloc[0]["sample.id"]
        if source_id not in source_wide.index:
            continue

        sink_counts = sink_wide.loc[sink_id]
        sources = {"External": source_wide.loc[source_id]}
        props = estimate_source_contributions(sink_counts, sources)

        for source_name, proportion in props.items():
            records.append({
                "sample.id": sink_id,
                carcass_col: sink_row_meta[carcass_col],
                time_col: sink_row_meta[time_col],
                "source": source_name,
                "proportion": proportion,
            })
    return pd.DataFrame.from_records(records)
