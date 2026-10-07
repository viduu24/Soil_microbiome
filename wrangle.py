# src/wrangle.py

import re
import pandas as pd
import numpy as np
from pathlib import Path

import diversity as div
import weather_merge as wm

DATA_DIR = Path("data")
OUT_DIR = Path("processed")
OUT_DIR.mkdir(exist_ok=True)

# Minimum library size (post-merge, pre-rarefaction total reads per sample)
# used to rarefy each rank's count table before computing relative
# abundance/diversity metrics. Kaszubinski et al. 2022 chose 7,000 (16S)
# and 4,000 (ITS) reads based on rarefaction curves rather than an
# arbitrary "smallest sample" cutoff -- inspect the rarefaction_curve()
# output (processed/*_rarefaction_curve.parquet) and adjust these before
# trusting downstream diversity/ANCOM results.
RAREFACTION_DEPTH = {"Family": 7000, "Phylum": 7000}

# Optional: source for the daily weather + cumulative ADH + observed
# Decomposition Stage log (one row per calendar day, shared across every
# carcass sampled that season). Fill this in to have wrangle.py attach a
# real "DaysSinceDeath" calendar-day count (and daily weather covariates) to
# every sample -- see attach_days_since_death() in weather_merge.py for why
# this can't just be parsed out of the sample.id naming convention.
#
# Two accepted formats:
#   - A single filepath (str) to one combined file that already has its own
#     "Season" column covering every season -- e.g.:
#       WEATHER_LOG_SOURCE = "C:/Users/vidus/OneDrive/Documents/Desktop/Benbow RA/Re_ Datasets/weather_log.txt"
#   - A dict {season_label: filepath} of separate per-season files, each
#     WITHOUT its own Season column -- e.g.:
#       WEATHER_LOG_SOURCE = {
#           "Fall": ".../fall_weather.txt", "Winter": ".../winter_weather.txt",
#           "Spring": ".../spring_weather.txt", "Summer": ".../summer_weather.txt",
#       }
#
# Leave this as None to skip the step entirely; the rest of the pipeline is
# unaffected either way.
WEATHER_LOG_SOURCE = "C:/Users/vidus/OneDrive/Documents/Desktop/Benbow RA/Re_ Datasets/dates_of_collection.xlsx"


def load_taxa_table(filepath: str, sep: str = "\t") -> pd.DataFrame:
    """
    Loads a taxa table where the first column is taxon name (header row)
    and subsequent rows are samples: sample.id, count, count, count...
    Returns a DataFrame indexed by sample.id, columns = taxon names.
    """
    df = pd.read_csv(filepath, sep=sep, header=0)
    df = df.rename(columns={df.columns[0]: "sample.id"})
    df = df.set_index("sample.id")
    return df


def load_metadata(filepath: str, sep: str = "\t") -> pd.DataFrame:
    meta = pd.read_csv(filepath, sep=sep)
    return meta


def filter_soil_samples(meta: pd.DataFrame) -> pd.DataFrame:
    """Keep only Soil samples. Adjust column name if it differs."""
    if "Order" not in meta.columns:
        raise ValueError("Expected an 'Order' column in metadata to filter on 'Soil'.")
    soil_meta = meta[meta["Order"].str.strip().str.lower() == "soil"].copy()
    return soil_meta


def merge_taxa_with_metadata(taxa_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join taxa counts with metadata on sample.id.
    Only samples present in both (and passing the Soil filter) survive.
    """
    merged = meta.merge(taxa_df, on="sample.id", how="inner")
    return merged


def condense_duplicate_taxa_columns(taxa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses columns that share the same base taxon name into a single
    summed column, and relabels blank/NA-derived names as "Unclassified".

    Why this is needed: taxa tables exported from ASV/OTU-level pipelines
    often assign multiple distinct sequence variants to the same Family or
    Phylum. When pandas reads a CSV with repeated header names (e.g.
    "Moraxellaceae" or "NA" appearing several times), it auto-appends
    ".1", ".2", ".3" etc. to make column labels unique. Left as-is, this
    fragments a single taxon's true total abundance across several
    separate columns/lines in downstream plots.

    IMPORTANT ORDERING NOTE: the NA/blank check must run on the
    suffix-stripped base name, not the raw column name. If a raw CSV
    header has "NA" repeated multiple times, only the *first* occurrence
    is literally "NA" -- pandas renames the rest to "NA.1", "NA.2", etc.
    at read time. Checking for exact "NA" before stripping the suffix (as
    an earlier version of this pipeline did via a separate
    recode_unclassified() pass) misses those renamed duplicates, causing
    them to collapse into their own stray "NA" taxon instead of being
    folded into "Unclassified". Stripping first, then checking, catches
    every variant in one pass.

    Must be called on the DataFrame indexed by sample.id (i.e. after
    load_taxa_table), and replaces the separate recode_unclassified step.
    """
    base_name_map = {}
    for col in taxa_df.columns:
        # Strip a trailing ".<one or more digits>" suffix, if present.
        stripped = re.sub(r"\.\d+$", "", str(col)).strip()

        # Recognize blank, literal "NA" (any case), and pandas' own
        # "Unnamed: N" placeholder (assigned when the raw CSV header was
        # blank) as unclassified taxa.
        if stripped == "" or stripped.upper() == "NA" or re.match(r"^Unnamed: \d+$", stripped):
            base_name_map[col] = "Unclassified"
        else:
            base_name_map[col] = stripped

    renamed = taxa_df.rename(columns=base_name_map)
    condensed = renamed.T.groupby(level=0).sum().T
    return condensed


def rarefy_merged(merged: pd.DataFrame, meta_cols: list[str], depth: int, seed: int = 0) -> pd.DataFrame:
    """
    Rarefies the taxon-count columns of a merged (meta + counts) wide
    table to `depth` total reads/sample (see diversity.rarefy_counts),
    dropping any sample below that depth, then re-attaches its metadata.
    Must run BEFORE compute_relative_abundance/melt_to_long so that
    downstream relative abundance, alpha/beta diversity, and ANCOM are
    all computed on a common sequencing depth across samples.
    """
    taxon_cols = [c for c in merged.columns if c not in meta_cols]
    counts = merged.set_index("sample.id")[taxon_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    rarefied_counts = div.rarefy_counts(counts, depth=depth, seed=seed)

    meta_indexed = merged.set_index("sample.id")[[c for c in meta_cols if c != "sample.id"]]
    rarefied_merged = meta_indexed.loc[rarefied_counts.index].join(rarefied_counts).reset_index()
    return rarefied_merged


def core_microbiome(long_df: pd.DataFrame, group_col: str, presence_threshold: float = 1.0) -> pd.DataFrame:
    """
    Identifies "core" taxa per group (e.g. Season, or an internal/external
    community label if you add one) -- taxa present (rel_abund > 0) in at
    least `presence_threshold` fraction of samples within that group,
    matching the core-microbiome framing used in both papers (shared
    taxa among internal vs. external, or among timepoints).

    Returns long-format: rank, group, taxon, prevalence (fraction of
    group's samples where the taxon is present), is_core.
    """
    records = []
    for (rank, group), sub in long_df.groupby(["rank", group_col]):
        n_samples = sub["sample.id"].nunique()
        prevalence = (
            sub[sub["rel_abund"] > 0]
            .groupby("taxon")["sample.id"]
            .nunique()
            .div(n_samples)
        )
        for taxon, prev in prevalence.items():
            records.append({
                "rank": rank,
                group_col: group,
                "taxon": taxon,
                "prevalence": prev,
                "is_core": prev >= presence_threshold,
            })
    return pd.DataFrame.from_records(records)


def compute_relative_abundance(merged: pd.DataFrame, meta_cols: list[str]) -> pd.DataFrame:
    """
    Given the merged wide table, compute relative abundance per sample
    across all taxon columns (i.e. every column not in meta_cols).
    """
    taxon_cols = [c for c in merged.columns if c not in meta_cols]
    counts = merged[taxon_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    totals = counts.sum(axis=1)
    rel_abund = counts.div(totals, axis=0)
    rel_abund.columns = [f"{c}_relabund" for c in rel_abund.columns]
    return pd.concat([merged[meta_cols], counts, rel_abund], axis=1)


def melt_to_long(wide: pd.DataFrame, meta_cols: list[str], rank_label: str) -> pd.DataFrame:
    """
    Converts wide relabund columns into long format:
    sample.id, <metadata...>, taxon, rel_abund, rank
    """
    relabund_cols = [c for c in wide.columns if c.endswith("_relabund")]
    long = wide.melt(
        id_vars=meta_cols,
        value_vars=relabund_cols,
        var_name="taxon",
        value_name="rel_abund",
    )
    long["taxon"] = long["taxon"].str.replace("_relabund", "", regex=False)
    long["rank"] = rank_label
    return long


def run_pipeline():
    meta_path = "C:/Users/vidus/OneDrive/Documents/Desktop/Benbow RA/Re_ Datasets/meta-ch3-new2.txt"
    family_path = "C:/Users/vidus/OneDrive/Documents/Desktop/Benbow RA/Re_ Datasets/Family_taxa_table.csv"
    phylum_path = "C:/Users/vidus/OneDrive/Documents/Desktop/Benbow RA/Re_ Datasets/Phylum_taxa_table.csv"

    # --- Load and filter metadata ---
    meta = load_metadata(meta_path)
    print("Order values:", meta["Order"].unique())

    soil_meta = filter_soil_samples(meta)
    print("Soil metadata rows:", len(soil_meta))

    # --- Optional: attach DaysSinceDeath + daily weather covariates ---
    # IMPORTANT: this must run BEFORE meta_cols is captured, so the new
    # columns (DaysSinceDeath, Date, weather variables) are correctly
    # recognized as metadata everywhere downstream -- not accidentally
    # treated as taxon abundance columns in compute_relative_abundance().
    if WEATHER_LOG_SOURCE:
        weather_log = wm.combine_weather_logs(WEATHER_LOG_SOURCE)
        soil_meta = wm.attach_days_since_death(soil_meta, weather_log)

        n_unresolved = soil_meta["DaysSinceDeath"].isna().sum()
        if n_unresolved > 0:
            unmatched_df = wm.diagnose_unmatched(soil_meta, weather_log)
            unmatched_path = OUT_DIR / "days_since_death_unmatched.csv"
            unmatched_df.to_csv(unmatched_path, index=False)
            print(f"attach_days_since_death: wrote {len(unmatched_df)} unmatched sample(s) "
                  f"with reasons to {unmatched_path}")

    meta_cols = list(soil_meta.columns)  # sample.id, Carcass, Family, Season, ADH, Stage, Order[, DaysSinceDeath, Date, weather...]

    # --- Load taxa tables (comma-separated) ---
    family_taxa = load_taxa_table(family_path, sep=",")
    phylum_taxa = load_taxa_table(phylum_path, sep=",")

    # --- Condense duplicate-named taxa columns (e.g. Moraxellaceae.1,
    # Moraxellaceae.19, Moraxellaceae.23 -> single summed "Moraxellaceae"
    # column; NA, NA.1, NA.2 -> single summed "Unclassified" column) ---
    n_family_cols_before = family_taxa.shape[1]
    n_phylum_cols_before = phylum_taxa.shape[1]
    family_taxa = condense_duplicate_taxa_columns(family_taxa)
    phylum_taxa = condense_duplicate_taxa_columns(phylum_taxa)
    print(f"Family taxa columns condensed: {n_family_cols_before} -> {family_taxa.shape[1]}")
    print(f"Phylum taxa columns condensed: {n_phylum_cols_before} -> {phylum_taxa.shape[1]}")

    family_taxa = family_taxa.reset_index()
    phylum_taxa = phylum_taxa.reset_index()

    # --- Sanity check: confirm sample.id overlap before merging ---
    print("Taxa sample.id sample:", family_taxa["sample.id"].head(5).tolist())
    print("Meta sample.id sample:", soil_meta["sample.id"].head(5).tolist())
    print("Overlap:", len(set(family_taxa["sample.id"]) & set(soil_meta["sample.id"])))

    # --- Merge ---
    family_merged = merge_taxa_with_metadata(family_taxa, soil_meta)
    phylum_merged = merge_taxa_with_metadata(phylum_taxa, soil_meta)

    # --- Rarefaction curves (saved for inspection -- use these to sanity
    # check RAREFACTION_DEPTH before trusting downstream diversity/ANCOM) ---
    for rank_label, merged, depth in [
        ("Family", family_merged, RAREFACTION_DEPTH["Family"]),
        ("Phylum", phylum_merged, RAREFACTION_DEPTH["Phylum"]),
    ]:
        counts_only = merged.set_index("sample.id")[[c for c in merged.columns if c not in meta_cols]]
        curve = div.rarefaction_curve(counts_only, steps=10)
        curve.to_parquet(OUT_DIR / f"{rank_label.lower()}_rarefaction_curve.parquet")
        n_below = (counts_only.sum(axis=1) < depth).sum()
        print(f"{rank_label}: {n_below}/{len(counts_only)} samples below "
              f"rarefaction depth {depth} and will be dropped.")

    # --- Rarefy to a common depth per rank before relative abundance ---
    family_merged = rarefy_merged(family_merged, meta_cols, RAREFACTION_DEPTH["Family"])
    phylum_merged = rarefy_merged(phylum_merged, meta_cols, RAREFACTION_DEPTH["Phylum"])

    # --- Compute relative abundance, melt to long format ---
    family_wide = compute_relative_abundance(family_merged, meta_cols)
    phylum_wide = compute_relative_abundance(phylum_merged, meta_cols)

    family_long = melt_to_long(family_wide, meta_cols, rank_label="Family")
    phylum_long = melt_to_long(phylum_wide, meta_cols, rank_label="Phylum")

    combined_long = pd.concat([family_long, phylum_long], ignore_index=True)

    # --- Alpha diversity per sample/rank (rarefied counts) ---
    alpha_frames = []
    for rank_label, merged in [("Family", family_merged), ("Phylum", phylum_merged)]:
        counts_only = merged.set_index("sample.id")[[c for c in merged.columns if c not in meta_cols]]
        a = div.alpha_diversity(counts_only)
        a["rank"] = rank_label
        alpha_frames.append(a.reset_index())
    alpha_df = pd.concat(alpha_frames, ignore_index=True)
    alpha_df = alpha_df.merge(soil_meta, on="sample.id", how="left")

    # --- Core microbiome per rank/Season (taxa present in ALL of a
    # Season's samples) -- change group_col/presence_threshold as needed ---
    core_df = core_microbiome(combined_long, group_col="Season", presence_threshold=1.0)

    # --- Save outputs ---
    family_wide.to_parquet(OUT_DIR / "family_wide.parquet")
    phylum_wide.to_parquet(OUT_DIR / "phylum_wide.parquet")
    combined_long.to_parquet(OUT_DIR / "combined_long.parquet")
    alpha_df.to_parquet(OUT_DIR / "alpha_diversity.parquet")
    core_df.to_parquet(OUT_DIR / "core_microbiome.parquet")

    print(f"Family samples after Soil filter + merge + rarefaction: {family_wide.shape[0]}")
    print(f"Phylum samples after Soil filter + merge + rarefaction: {phylum_wide.shape[0]}")
    print(f"Combined long table shape: {combined_long.shape}")

    return family_wide, phylum_wide, combined_long


if __name__ == "__main__":
    run_pipeline()