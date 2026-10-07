# src/weather_merge.py
"""
Merges a daily weather + cumulative ADH + observed decomposition Stage log
onto the sample metadata, to attach a real calendar-day "DaysSinceDeath"
count (and the underlying daily weather covariates) to each sample.

Why this is needed: the numeric segment in sample.id (e.g. the "28" in
C7-28-S) is a naming/sequence index, NOT literal elapsed calendar days --
cross-checking against the actual daily weather log shows samples labeled
"-28-" can correspond to as few as ~12-13 real elapsed days, since ADH
(temperature-accumulated, not calendar-linear) is what actually governs
decomposition progress. Real elapsed days have to be derived from the
weather log's Date column instead.
"""

import pandas as pd


WEATHER_COLUMN_RENAME = {
    "Decomposition": "Stage",
    "Decomposition Stage": "Stage",
}


def _add_days_since_death(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds DaysSinceDeath = elapsed calendar days since the EARLIEST logged
    date, computed WITHIN EACH SEASON separately (via groupby) -- each
    season has its own placement date, so day-zero must never be computed
    globally across seasons, only per season.
    """
    df = df.sort_values(["Season", "Date"]).reset_index(drop=True)
    df["DaysSinceDeath"] = df.groupby("Season")["Date"].transform(lambda s: (s - s.min()).dt.days)
    return df


def _read_table_any_format(filepath: str, sep: str) -> pd.DataFrame:
    """
    Reads either an Excel file (.xlsx/.xls) or a delimited text file
    (.txt/.csv/.tsv), auto-detected from the file extension, so callers
    don't need to convert Excel exports to CSV/TSV by hand first.
    """
    suffix = str(filepath).lower().rsplit(".", 1)[-1]
    if suffix in ("xlsx", "xls"):
        return pd.read_excel(filepath)
    return pd.read_csv(filepath, sep=sep)


def _normalize_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes Season values to Title Case (e.g. "spring" -> "Spring") so
    a casing difference between the metadata and the weather log doesn't
    silently break every (Season, ADH, Stage) match for that season. This
    is safe to do unconditionally -- "Spring" and "spring" are never
    meant to be different seasons.
    """
    before = df["Season"].astype(str).str.strip()
    after = before.str.title()
    changed = (before != after) & before.notna()
    if changed.any():
        examples = sorted(before[changed].unique().tolist())
        print(f"load_weather_log: normalized Season casing for {changed.sum()} row(s) "
              f"(e.g. {examples[:5]} -> title case) so casing differences don't break matching.")
    df = df.copy()
    df["Season"] = after
    return df


def _clean_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans "X | Y"-style Stage values (an apparent formula/copy-paste
    artifact seen in some weather logs, concatenating two values with a
    pipe). Segments that are identical once stripped (e.g. "Active |
    Active") are safely collapsed to the single value -- that's
    unambiguous. Segments that genuinely DIFFER (e.g. "Advanced |
    Active", "Dry | Active") are NOT silently resolved: some of these
    combinations go backward relative to normal decomposition order,
    so guessing which half is "correct" risks quietly corrupting real
    Stage data. Those rows keep their raw, uncleaned value (so they'll
    fail to match downstream and show up in attach_days_since_death's
    diagnostics) and are printed here so they can be fixed by hand in
    the source file.
    """
    df = df.copy()
    raw = df["Stage"].astype(str).str.strip()
    df["Stage"] = raw  # always apply the whitespace strip, pipes or not
    has_pipe = raw.str.contains(r"\|", na=False)
    if not has_pipe.any():
        return df

    def resolve(val: str):
        parts = [p.strip() for p in val.split("|")]
        parts = [p for p in parts if p]  # drop empty segments (e.g. "| Advanced")
        unique_parts = set(parts)
        if len(unique_parts) <= 1:
            return parts[0] if parts else val
        return val  # genuinely conflicting -- leave as-is, flagged below

    cleaned = raw.where(~has_pipe, raw[has_pipe].apply(resolve))
    still_conflicting = has_pipe & (cleaned == raw) & raw.str.contains(r"\|", na=False)
    resolved_count = int((has_pipe & ~still_conflicting).sum())
    if resolved_count > 0:
        print(f"load_weather_log: cleaned {resolved_count} Stage value(s) of the form "
              f"'X | X' (identical duplicate) down to just 'X'.")
    if still_conflicting.any():
        conflicts = sorted(raw[still_conflicting].unique().tolist())
        print(f"load_weather_log: {still_conflicting.sum()} Stage value(s) contain a '|' with "
              f"GENUINELY DIFFERING values on each side -- left unresolved rather than guessed, "
              f"since some combinations (e.g. Advanced|Active) go backward relative to normal "
              f"decomposition order and picking the wrong side could silently corrupt Stage data. "
              f"These rows will fail to match and need manual fixing in the source file: {conflicts}")
    df["Stage"] = cleaned
    return df


def load_weather_log(filepath: str, season_label: str = None, sep: str = "\t") -> pd.DataFrame:
    """
    Loads a daily weather/ADH/Stage log. Accepts .xlsx/.xls (Excel) or a
    delimited text file (.txt/.csv/.tsv, using `sep`) -- format is
    auto-detected from the file extension.

    - If `season_label` is given, that season is applied to every row --
      use this for a file covering just ONE season with no Season column
      of its own.
    - If `season_label` is None (default), the file must already contain
      its own "Season" column -- use this for a single combined file that
      covers every season together (as produced by, e.g., exporting one
      combined weather/ADH/Stage table with a Season column included).
    """
    df = _read_table_any_format(filepath, sep)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=WEATHER_COLUMN_RENAME)
    df["Date"] = pd.to_datetime(df["Date"])

    if season_label is not None:
        df["Season"] = season_label
    elif "Season" not in df.columns:
        raise ValueError(
            "load_weather_log: no season_label was given and no 'Season' column "
            "was found in the file -- either pass season_label='Fall' (etc.) for a "
            "single-season file, or make sure the file already has its own Season column."
        )

    df = _normalize_season(df)
    if "Stage" in df.columns:
        df = _clean_stage(df)

    return _add_days_since_death(df)


def combine_weather_logs(source, sep: str = "\t") -> pd.DataFrame:
    """
    Loads the daily weather/ADH/Stage log(s). `source` can be:
      - a single filepath (str) to ONE combined file that already has its
        own Season column covering every season, or
      - a dict {season_label: filepath} of separate per-season files
        (each without its own Season column).
    """
    if isinstance(source, dict):
        frames = [load_weather_log(fp, season_label=season, sep=sep) for season, fp in source.items()]
        return pd.concat(frames, ignore_index=True)
    return load_weather_log(source, season_label=None, sep=sep)


def attach_days_since_death(
    meta: pd.DataFrame,
    weather_log: pd.DataFrame,
    adh_tolerance: float = 5.0,
) -> pd.DataFrame:
    """
    Merges DaysSinceDeath (+ daily weather covariates) onto sample metadata,
    matching on (Season, ADH, Stage). Because ADH/Stage are recorded at the
    season level (identical across every carcass sampled that day), this is
    a broadcast join: many metadata rows sharing the same Season/ADH/Stage
    correctly match the same single weather-log row.

    Small ADH rounding differences between the metadata and the weather log
    (e.g. metadata says 844, the log's nearest value is 843) mean an exact
    match can silently miss real matches near stage transitions. To avoid
    silently dropping those samples, unmatched rows fall back to the
    nearest ADH value within the same (Season, Stage), but only if within
    `adh_tolerance` -- and every fallback is reported, not hidden.
    """
    weather_cols = [c for c in [
        "Season", "ADH", "Stage", "Date", "DaysSinceDeath",
        "Max Air Temp", "Min Air Temp", "Mean Air Temp", "Precipitation",
        "Max Relative Humidity", "Min Relative Humidity", "Mean Relative Humidity",
        "Max Wind Speed",
    ] if c in weather_log.columns]

    meta = meta.copy()
    meta["ADH"] = meta["ADH"].astype(float)
    weather_log = weather_log.copy()
    weather_log["ADH"] = weather_log["ADH"].astype(float)

    # ADH can plateau across multiple consecutive calendar days (e.g. a cold
    # snap that adds ~0 degree-hours), so (Season, ADH, Stage) is not
    # guaranteed unique in the weather log. Left un-deduplicated, a single
    # metadata row matching a repeated ADH value would silently fan out into
    # one duplicate output row per matching date -- inflating the sample
    # count. We keep the EARLIEST date for each (Season, ADH, Stage) group,
    # since ADH alone can't disambiguate which day it really was, and report
    # how many groups this affected so it isn't a silent decision.
    dup_counts = weather_log.groupby(["Season", "ADH", "Stage"]).size()
    n_ambiguous_groups = int((dup_counts > 1).sum())
    if n_ambiguous_groups > 0:
        print(f"attach_days_since_death: {n_ambiguous_groups} (Season, ADH, Stage) "
              f"combination(s) in the weather log correspond to more than one calendar "
              f"date (ADH plateaued) -- keeping the earliest date for each.")
    weather_log = (
        weather_log.sort_values("Date")
        .drop_duplicates(subset=["Season", "ADH", "Stage"], keep="first")
    )

    merged = meta.merge(weather_log[weather_cols], on=["Season", "ADH", "Stage"], how="left", indicator=True)
    exact_mask = merged["_merge"] == "both"
    n_exact = int(exact_mask.sum())
    matched = merged[exact_mask].drop(columns="_merge")
    unmatched = merged[~exact_mask].drop(columns="_merge")

    fill_cols = [c for c in weather_cols if c not in ("Season", "ADH", "Stage")]
    unmatched = unmatched.drop(columns=fill_cols)

    n_fallback = 0
    n_unresolved = 0
    no_category_match = []  # (Season, Stage) pairs with zero weather-log rows at all
    adh_too_far = []        # rows where the category matched but ADH gap exceeded tolerance
    fallback_rows = []
    for _, row in unmatched.iterrows():
        candidates = weather_log[
            (weather_log["Season"] == row["Season"]) & (weather_log["Stage"] == row["Stage"])
        ]
        if candidates.empty:
            n_unresolved += 1
            no_category_match.append((row["Season"], row["Stage"]))
            fallback_rows.append(row)
            continue
        best_idx = (candidates["ADH"] - row["ADH"]).abs().idxmin()
        best = candidates.loc[best_idx]
        gap = abs(best["ADH"] - row["ADH"])
        if gap <= adh_tolerance:
            for c in fill_cols:
                row[c] = best[c]
            n_fallback += 1
        else:
            n_unresolved += 1
            adh_too_far.append((row["Season"], row["Stage"], row["ADH"], best["ADH"], gap))
        fallback_rows.append(row)

    unmatched = pd.DataFrame(fallback_rows) if fallback_rows else unmatched
    result = pd.concat([matched, unmatched], ignore_index=True, sort=False)

    print(f"attach_days_since_death: {n_exact} exact (Season, ADH, Stage) matches, "
          f"{n_fallback} matched via nearest-ADH fallback (tolerance={adh_tolerance}), "
          f"{n_unresolved} could not be matched at all.")

    if no_category_match:
        unique_missing = sorted(set(no_category_match))
        print(f"attach_days_since_death: {len(no_category_match)} unresolved sample(s) had NO "
              f"(Season, Stage) match in the weather log at all -- likely a spelling/formatting "
              f"mismatch between the two files. Missing (Season, Stage) combinations:")
        for season, stage in unique_missing:
            print(f"    metadata has Season={season!r}, Stage={stage!r} -- not found in weather log")
        meta_seasons = sorted(meta["Season"].dropna().unique().tolist())
        meta_stages = sorted(meta["Stage"].dropna().unique().tolist())
        weather_seasons = sorted(weather_log["Season"].dropna().unique().tolist())
        weather_stages = sorted(weather_log["Stage"].dropna().unique().tolist())
        print(f"    metadata Season values: {meta_seasons}")
        print(f"    weather log Season values: {weather_seasons}")
        print(f"    metadata Stage values: {meta_stages}")
        print(f"    weather log Stage values: {weather_stages}")

    if adh_too_far:
        print(f"attach_days_since_death: {len(adh_too_far)} unresolved sample(s) matched on "
              f"(Season, Stage) but had no weather-log ADH within tolerance={adh_tolerance}. "
              f"Largest gaps:")
        for season, stage, meta_adh, nearest_adh, gap in sorted(adh_too_far, key=lambda r: -r[4])[:5]:
            print(f"    Season={season!r}, Stage={stage!r}: metadata ADH={meta_adh}, "
                  f"nearest weather-log ADH={nearest_adh} (gap={gap:.1f})")

    return result


def diagnose_unmatched(result: pd.DataFrame, weather_log: pd.DataFrame, adh_tolerance: float = 5.0) -> pd.DataFrame:
    """
    Given the DataFrame returned by attach_days_since_death(), returns a
    tidy table of every sample that failed to get a DaysSinceDeath value,
    with the specific reason for each:

      - "no (Season, Stage) match" -- the weather log has no rows at all
        for that Season+Stage combination (spelling/formatting mismatch,
        or a genuinely conflicting Stage value like "Advanced | Active"
        that was deliberately left unresolved).
      - "ADH gap too large" -- Season+Stage matched, but the nearest
        available weather-log ADH was farther than `adh_tolerance` away
        (usually means the weather log has a date-range gap).

    Columns: sample.id, Season, Stage, ADH, reason, nearest_available_adh,
    adh_gap. Use this to get the exact list of affected sample.id's
    rather than just the aggregate counts printed by
    attach_days_since_death().
    """
    unresolved = result[result["DaysSinceDeath"].isna()].copy()
    if unresolved.empty:
        return pd.DataFrame(columns=["sample.id", "Season", "Stage", "ADH", "reason",
                                      "nearest_available_adh", "adh_gap"])

    records = []
    for _, row in unresolved.iterrows():
        candidates = weather_log[
            (weather_log["Season"] == row["Season"]) & (weather_log["Stage"] == row["Stage"])
        ]
        if candidates.empty:
            records.append({
                "sample.id": row["sample.id"], "Season": row["Season"], "Stage": row["Stage"],
                "ADH": row["ADH"], "reason": "no (Season, Stage) match in weather log",
                "nearest_available_adh": None, "adh_gap": None,
            })
        else:
            best_idx = (candidates["ADH"] - row["ADH"]).abs().idxmin()
            best_adh = candidates.loc[best_idx, "ADH"]
            gap = abs(best_adh - row["ADH"])
            records.append({
                "sample.id": row["sample.id"], "Season": row["Season"], "Stage": row["Stage"],
                "ADH": row["ADH"], "reason": f"ADH gap too large (tolerance={adh_tolerance})",
                "nearest_available_adh": best_adh, "adh_gap": gap,
            })
    return pd.DataFrame.from_records(records).sort_values(["reason", "Season", "Stage", "ADH"])