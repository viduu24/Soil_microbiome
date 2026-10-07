# src/upload_pipeline.py
"""
Generalized version of wrangle.py's run_pipeline(), for researchers using
their OWN dataset through the "Upload Data" page rather than editing
wrangle.py and running it from the command line.

Reuses every actual analysis function from wrangle.py / diversity.py /
weather_merge.py unchanged (condensing duplicate taxa columns,
rarefaction, relative abundance, melting to long format, alpha diversity,
core microbiome, weather/DaysSinceDeath merge) -- the only thing this
module adds is a layer that:
  1. Accepts uploaded file objects instead of hardcoded local paths.
  2. Accepts a user-supplied column mapping (their own column names ->
     the canonical names the pipeline functions expect: sample.id,
     Season, ADH, Stage, plus optional Family/position and Carcass), and
     renames columns accordingly before running the same pipeline logic.
  3. Returns results as DataFrames in memory, rather than writing parquet
     files to disk -- see data_source.py for how the dashboard pages then
     read whichever source (uploaded or local files) is active.
"""

import pandas as pd

import wrangle as w
import diversity as div
import weather_merge as wm


def _reset_stream(file) -> None:
    """
    Resets a file-like object's read position to the start, if it supports
    seeking. Needed because the SAME uploaded file object often gets read
    more than once (e.g. once for a preview table, again inside the actual
    pipeline) -- without seeking back to 0, the second read sees an
    already-exhausted stream and fails (e.g. "Could not determine
    delimiter" from an apparently-empty file). Plain path strings don't
    have `.seek`, so this is a no-op for those.
    """
    if hasattr(file, "seek"):
        file.seek(0)


def read_metadata_any_format(file, sep: str = None) -> pd.DataFrame:
    """
    Reads a metadata file with a flexible delimiter -- unlike wrangle.py's
    load_metadata() (which assumes tab-separated, since that's this
    project's own file format), an uploaded file from another researcher
    could be comma, tab, or semicolon separated. If `sep` isn't given,
    pandas' python engine auto-detects it from the file content.
    """
    _reset_stream(file)
    filename = getattr(file, "name", str(file))
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    if sep is not None:
        return pd.read_csv(file, sep=sep)
    return pd.read_csv(file, sep=None, engine="python")


def apply_column_mapping(meta: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Renames the researcher's own column names to the canonical names the
    rest of the pipeline expects, based on a mapping dict like:
        {"sample_id": "SampleID", "season": "collection_season",
         "adh": "DegreeHours", "stage": "decomp_stage",
         "position": "swab_location", "carcass": "subject_id"}
    Only sample_id, season, adh, and stage are required; position and
    carcass are optional (position enables the same Side/Under-style
    grouping used throughout the dashboard; carcass enables Source
    Tracking, which orders samples per individual over time).
    """
    canonical_names = {
        "sample_id": "sample.id", "season": "Season", "adh": "ADH",
        "stage": "Stage", "position": "Family", "carcass": "Carcass",
    }
    rename_map = {mapping[key]: canonical for key, canonical in canonical_names.items()
                  if key in mapping and mapping[key]}
    return meta.rename(columns=rename_map)


def run_upload_pipeline(
    metadata_file,
    family_taxa_file,
    phylum_taxa_file,
    column_mapping: dict,
    filter_col: str = None,
    filter_value: str = None,
    rarefaction_depth: dict = None,
    metadata_sep: str = None,
    taxa_sep: str = ",",
    core_group_col: str = "Season",
) -> dict:
    """
    Runs the full wrangling pipeline (condense duplicate taxa columns,
    filter, merge, rarefy, relative abundance, melt to long, alpha
    diversity, core microbiome) on uploaded files with a user-supplied
    column mapping, mirroring wrangle.py's run_pipeline() but generalized.

    `family_taxa_file` / `phylum_taxa_file` are each optional (pass None
    to skip that rank) -- at least one must be provided.
    `filter_col` / `filter_value`: optional row filter on the ORIGINAL
    (pre-rename) metadata column name, e.g. filter_col="Order",
    filter_value="Soil" -- pass filter_col=None to skip filtering
    entirely (use every row).
    `rarefaction_depth`: dict like {"Family": 7000, "Phylum": 7000};
    a rank is skipped if its key is missing or None (no rarefaction,
    relative abundance computed on raw counts -- not recommended, but
    available if the researcher's data is already normalized).

    Returns a dict: combined_long, family_wide, phylum_wide, alpha_diversity,
    core_microbiome, rarefaction_curves (dict of rank -> curve DataFrame,
    for the researcher to sanity-check their chosen depth), warnings (list
    of str, e.g. samples dropped at each step).
    """
    if family_taxa_file is None and phylum_taxa_file is None:
        raise ValueError("At least one of family_taxa_file / phylum_taxa_file must be provided.")

    warnings = []
    meta = read_metadata_any_format(metadata_file, sep=metadata_sep)

    if filter_col:
        if filter_col not in meta.columns:
            raise ValueError(f"Filter column '{filter_col}' not found in metadata columns: {list(meta.columns)}")
        before = len(meta)
        meta = meta[meta[filter_col].astype(str).str.strip().str.lower() == str(filter_value).strip().lower()].copy()
        warnings.append(f"Filtered '{filter_col}' == '{filter_value}': {before} -> {len(meta)} rows.")

    meta = apply_column_mapping(meta, column_mapping)
    required = ["sample.id", "Season", "ADH", "Stage"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Required column(s) missing after mapping: {missing}. "
                          f"Check your column mapping selections.")

    meta_cols = list(meta.columns)
    rarefaction_depth = rarefaction_depth or {}

    rank_inputs = [("Family", family_taxa_file), ("Phylum", phylum_taxa_file)]
    wide_by_rank, long_by_rank, alpha_by_rank, curves_by_rank = {}, {}, {}, {}

    for rank_label, taxa_file in rank_inputs:
        if taxa_file is None:
            continue

        _reset_stream(taxa_file)
        taxa = pd.read_csv(taxa_file, sep=taxa_sep)
        taxa = taxa.rename(columns={taxa.columns[0]: "sample.id"}).set_index("sample.id")
        n_cols_before = taxa.shape[1]
        taxa = w.condense_duplicate_taxa_columns(taxa)
        warnings.append(f"{rank_label}: condensed {n_cols_before} -> {taxa.shape[1]} taxa columns.")
        taxa = taxa.reset_index()

        merged = w.merge_taxa_with_metadata(taxa, meta)
        overlap = merged["sample.id"].nunique()
        warnings.append(f"{rank_label}: {overlap} samples matched between metadata and taxa table.")
        if overlap == 0:
            raise ValueError(
                f"{rank_label}: no sample.id values matched between the metadata and taxa "
                f"table -- check that your 'Sample ID' column mapping is correct and that "
                f"the IDs are formatted the same way in both files."
            )

        counts_only = merged.set_index("sample.id")[[c for c in merged.columns if c not in meta_cols]]
        curve = div.rarefaction_curve(counts_only, steps=10)
        curves_by_rank[rank_label] = curve

        depth = rarefaction_depth.get(rank_label)
        if depth:
            n_before = len(merged)
            merged = w.rarefy_merged(merged, meta_cols, depth)
            warnings.append(f"{rank_label}: rarefied to depth {depth} -- {n_before} -> {len(merged)} samples "
                             f"(dropped samples below depth).")
        else:
            warnings.append(f"{rank_label}: no rarefaction depth set -- using raw (unrarefied) counts.")

        wide = w.compute_relative_abundance(merged, meta_cols)
        wide_by_rank[rank_label] = wide
        long_by_rank[rank_label] = w.melt_to_long(wide, meta_cols, rank_label=rank_label)

        counts_final = merged.set_index("sample.id")[[c for c in merged.columns if c not in meta_cols]]
        alpha = div.alpha_diversity(counts_final)
        alpha["rank"] = rank_label
        alpha_by_rank[rank_label] = alpha.reset_index()

    combined_long = pd.concat(list(long_by_rank.values()), ignore_index=True)
    alpha_df = pd.concat(list(alpha_by_rank.values()), ignore_index=True).merge(meta, on="sample.id", how="left")

    core_df = None
    if core_group_col in meta_cols:
        core_df = w.core_microbiome(combined_long, group_col=core_group_col, presence_threshold=1.0)
    else:
        warnings.append(f"Skipped core microbiome: '{core_group_col}' not present in mapped metadata.")

    return {
        "combined_long": combined_long,
        "family_wide": wide_by_rank.get("Family"),
        "phylum_wide": wide_by_rank.get("Phylum"),
        "alpha_diversity": alpha_df,
        "core_microbiome": core_df,
        "rarefaction_curves": curves_by_rank,
        "warnings": warnings,
    }


def run_upload_weather_merge(
    combined_long: pd.DataFrame,
    alpha_diversity: pd.DataFrame,
    weather_file,
    weather_column_mapping: dict,
    weather_sep: str = None,
    adh_tolerance: float = 5.0,
) -> dict:
    """
    Optional second step: merges a daily weather/ADH/Stage log onto
    already-processed results (from run_upload_pipeline), attaching
    DaysSinceDeath the same way wrangle.py's WEATHER_LOG_SOURCE does.

    `weather_column_mapping`: e.g. {"date": "CollectionDate", "adh": "ADD",
    "stage": "VisualStage", "season": "Season"} -- season is optional; if
    omitted, the weather file's Season column (if present) is used as-is,
    or you must pass season_label separately for a single-season file
    (not currently exposed here -- combined multi-season files only).

    Returns updated combined_long / alpha_diversity (with DaysSinceDeath
    and weather covariates attached) plus the same kind of `warnings` list.
    """
    _reset_stream(weather_file)
    filename = getattr(weather_file, "name", str(weather_file))
    weather_raw = pd.read_excel(weather_file) if filename.lower().endswith((".xlsx", ".xls")) else (
        pd.read_csv(weather_file, sep=weather_sep) if weather_sep else pd.read_csv(weather_file, sep=None, engine="python")
    )

    rename_map = {}
    if "date" in weather_column_mapping:
        rename_map[weather_column_mapping["date"]] = "Date"
    if "adh" in weather_column_mapping:
        rename_map[weather_column_mapping["adh"]] = "ADH"
    if "stage" in weather_column_mapping:
        rename_map[weather_column_mapping["stage"]] = "Stage"
    if "season" in weather_column_mapping and weather_column_mapping["season"]:
        rename_map[weather_column_mapping["season"]] = "Season"
    weather_raw = weather_raw.rename(columns=rename_map)

    weather_raw["Date"] = pd.to_datetime(weather_raw["Date"])
    if "Season" not in weather_raw.columns:
        raise ValueError("Weather file needs a Season column (or map one) to match multi-season metadata.")

    weather_log = wm._normalize_season(weather_raw)
    if "Stage" in weather_log.columns:
        weather_log = wm._clean_stage(weather_log)
    weather_log = wm._add_days_since_death(weather_log)

    combined_long = wm.attach_days_since_death(combined_long, weather_log, adh_tolerance=adh_tolerance)
    alpha_diversity = wm.attach_days_since_death(alpha_diversity, weather_log, adh_tolerance=adh_tolerance)

    n_unresolved = int(combined_long["DaysSinceDeath"].isna().sum())
    warnings = [f"{len(combined_long) - n_unresolved}/{len(combined_long)} combined_long rows matched a weather date."]

    return {"combined_long": combined_long, "alpha_diversity": alpha_diversity, "warnings": warnings}
