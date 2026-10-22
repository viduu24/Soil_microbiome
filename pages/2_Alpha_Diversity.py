# src/alpha_diversity_page.py
"""
Streamlit page: alpha diversity (observed richness, Chao1, Shannon) on
rarefied counts, viewable by any combination of metadata parameters, month by
month, or by actual sampling date. Computation comes from diversity.py; this
file is UI only.

Run with:  python -m streamlit run alpha_diversity_page.py
"""

import sys
from pathlib import Path

import copy

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from plotly.offline import get_plotlyjs_version
from scipy.stats import kruskal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diversity import (rarefy_counts, alpha_diversity, observed_richness,  # noqa: E402
                       shannon_diversity)

# ---------------------------------------------------------------------------
# ADAPT THESE to match what wrangle.py writes
# ---------------------------------------------------------------------------
PARQUET_NAME = "combined_long.parquet"   # the long-format parquet from wrangle.py


def _find_processed_dir() -> Path:
    """Walk up from this file until a processed/ folder containing the parquet
    is found (works from src/ or src/soil_microbiome_pipeline/pages/)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "processed" / PARQUET_NAME).exists():
            return parent / "processed"
    return Path(__file__).resolve().parent / "processed"


PROCESSED_DIR = _find_processed_dir()
SAMPLE_COL = "sample.id"
TAXON_COL = "taxon"                      # column holding the taxon name
COUNT_COL = "count"                      # raw read counts; if absent, rel_abund is used instead
REL_COL = "rel_abund"
LEVEL_COL = "rank"                       # Family / Phylum; set to None if absent
META_COLS = ["Carcass", "Family", "Season", "Stage", "ADH"]  # "Family" (Under/Side) is shown as Location
COLOR_SEQ = px.colors.qualitative.Plotly  # same colors on screen and in the download
# Month is taken from a "Month" column if you have one, otherwise derived
# from the date column chosen in the sidebar (default: first column whose name
# contains "date", case-insensitive). The "By sampling date" view always uses
# that date column.

METRICS = {"observed_richness": "Observed richness", "chao1": "Chao1", "shannon": "Shannon"}
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DATE_FORMATS = ["Auto", "Day first (DD/MM/YYYY)", "Month first (MM/DD/YYYY)"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_long() -> pd.DataFrame:
    path = PROCESSED_DIR / PARQUET_NAME
    if not path.exists():
        found = sorted(PROCESSED_DIR.glob("*.parquet"))
        if len(found) == 1:
            path = found[0]
        else:
            st.error(f"Couldn't find {PARQUET_NAME} in {PROCESSED_DIR}. "
                     f"Parquet files there: {[p.name for p in found]}. "
                     "Set PARQUET_NAME at the top of this file.")
            st.stop()
    df = pd.read_parquet(path)
    missing = [c for c in (SAMPLE_COL, TAXON_COL) if c not in df.columns]
    if COUNT_COL not in df.columns and REL_COL not in df.columns:
        missing.append(f"{COUNT_COL} or {REL_COL}")
    if missing:
        st.error(f"Missing columns {missing}. Columns found: {list(df.columns)}")
        st.stop()
    return df


_DATE_RE = r"^\s*(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})"


def _infer_order(text: pd.Series):
    """Look at numeric dates like 13/04/2025 and work out the order from unambiguous values.
    Returns "iso" (year first), "day", "month", or None when it can't be told."""
    parts = text.str.extract(_DATE_RE).dropna().astype(int)
    if parts.empty:
        return None
    if (parts[0] > 31).any():
        return "iso"
    day_evidence, month_evidence = (parts[0] > 12).any(), (parts[1] > 12).any()
    if day_evidence and not month_evidence:
        return "day"
    if month_evidence and not day_evidence:
        return "month"
    return None


def date_is_ambiguous(s: pd.Series) -> bool:
    """True for text dates like 05/04/2025 where every day and month number is 12 or less."""
    if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return False
    text = s.dropna().astype(str).str.strip()
    return (not text.empty and text.str.match(_DATE_RE).any() and _infer_order(text) is None)


def _to_datetime(s: pd.Series, dayfirst: bool) -> pd.Series:
    """Parse text dates one by one (so mixed formats still parse), bad values become NaT."""
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")
    except (ValueError, TypeError):   # older pandas without format="mixed"
        return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)


def parse_dates(s: pd.Series, fmt: str = "Auto") -> pd.Series:
    """Robust date parser. Handles real datetimes, Excel serial numbers and text dates.
    In Auto mode the day/month order is worked out from the data (any value with a day
    above 12 settles it); fully ambiguous dates are read month first, and the sidebar says so."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s
    if pd.api.types.is_numeric_dtype(s) and s.dropna().between(20000, 80000).all():
        return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")   # Excel serial dates
    text = s.astype(str).str.strip()
    order = _infer_order(text)
    if order == "iso":
        dayfirst = False                      # 2025-04-05 is unambiguous
    elif fmt.startswith("Day"):
        dayfirst = True
    elif fmt.startswith("Month"):
        dayfirst = False
    else:
        dayfirst = order == "day"
    return _to_datetime(text, dayfirst)


def add_month(meta: pd.DataFrame, date_col=None, date_fmt: str = "Auto"):
    """Adds a 'Month' column (ordered chronologically) and a 'SampleDate' column (actual
    day). Returns (meta, ordered_month_labels) or (meta, None) if no month information exists."""
    if date_col is None:
        date_col = next((c for c in meta.columns if "date" in str(c).lower()), None)
    dt = None
    if date_col is not None and date_col in meta.columns:
        dt = parse_dates(meta[date_col], date_fmt)
        meta = meta.assign(SampleDate=dt.dt.normalize())

    if "Month" in meta.columns:
        raw = meta["Month"]
        if pd.api.types.is_numeric_dtype(raw):
            num = raw.astype("Int64")
        else:
            lookup = {m.lower(): i + 1 for i, m in enumerate(MONTH_ABBR)}
            num = raw.astype(str).str[:3].str.lower().map(lookup).astype("Int64")
        meta = meta.assign(Month=num.map(lambda n: MONTH_ABBR[n - 1] if pd.notna(n) else None))
        return meta, [m for m in MONTH_ABBR if m in set(meta["Month"].dropna())]

    if dt is None:
        return meta, None
    multi_year = dt.dt.year.nunique() > 1
    labels = dt.dt.strftime("%Y-%m") if multi_year else dt.dt.strftime("%b")
    meta = meta.assign(Month=labels.where(dt.notna()))
    order = (sorted(labels.dropna().unique()) if multi_year
             else [m for m in MONTH_ABBR if m in set(labels.dropna())])
    return meta, order


@st.cache_data(show_spinner="Rarefying and computing alpha diversity...")
def compute_alpha(long_df: pd.DataFrame, depth) -> pd.DataFrame:
    if depth is None:  # no raw counts available: work from relative abundance
        wide = long_df.pivot_table(index=SAMPLE_COL, columns=TAXON_COL, values=REL_COL,
                                   aggfunc="sum", fill_value=0)
        return pd.DataFrame({"observed_richness": observed_richness(wide),
                             "shannon": shannon_diversity(wide)})
    wide = long_df.pivot_table(index=SAMPLE_COL, columns=TAXON_COL, values=COUNT_COL,
                               aggfunc="sum", fill_value=0)
    return alpha_diversity(rarefy_counts(wide, depth))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Alpha diversity", layout="wide")
st.title("Alpha diversity")

df = load_long()
date_like = [c for c in df.columns if "date" in str(c).lower()]

with st.sidebar:
    st.header("Data")
    if LEVEL_COL and LEVEL_COL in df.columns:
        level = st.selectbox("Taxonomic level", sorted(r for r in df[LEVEL_COL].dropna().unique()
                                                       if str(r).strip().casefold() != "order"))
        df = df[df[LEVEL_COL] == level]

    has_counts = COUNT_COL in df.columns
    if has_counts:
        totals = df.groupby(SAMPLE_COL)[COUNT_COL].sum()
        depth = st.slider("Rarefaction depth (reads per sample)", int(totals.min()),
                          int(totals.max()), int(totals.quantile(0.1)))
        st.caption(f"{(totals >= depth).sum()} of {len(totals)} samples kept at this depth.")
        available = METRICS
    else:
        depth = None
        available = {k: v for k, v in METRICS.items() if k != "chao1"}
        st.info("No raw counts in the parquet, so these come from relative abundance: "
                "no rarefaction and no Chao1.")

    metric = st.radio("Alpha diversity metric", list(available), format_func=METRICS.get)
    metrics = [metric]

    date_col, date_fmt = None, "Auto"
    if date_like:
        st.header("Dates")
        date_col = st.selectbox("Date column", date_like,
                                help="Used for the Month by month and By sampling date views.")
        date_fmt = st.radio("Date format", DATE_FORMATS,
                            help="Auto works the order out from the data (any date with a day above 12 "
                                 "settles it). If days and months look swapped, choose Day first.")
        if date_fmt == "Auto" and date_is_ambiguous(df[date_col]):
            st.caption("Every day and month number in this column is 12 or less, so the order can't be "
                       "detected: dates are being read month first. If days and months look swapped "
                       "(e.g. 5 Apr shown as 4 May), choose Day first above.")

alpha = compute_alpha(df, depth)
meta = (df[[SAMPLE_COL] + [c for c in META_COLS + ["Month"] + date_like if c in df.columns]]
        .drop_duplicates(SAMPLE_COL).set_index(SAMPLE_COL))
meta = meta.rename(columns={"Family": "Location"})
meta, month_order = add_month(meta, date_col, date_fmt)
data = alpha.join(meta, how="inner")

params = [c for c in meta.columns if c != "ADH" and c not in date_like and c != "SampleDate"]
if not metrics or not params:
    st.info("Pick at least one metric.")
    st.stop()

view = st.radio("View", ["By parameters", "Month by month", "By sampling date"], horizontal=True)

mode = None
date_summary = None
show_all_dates = False
if view == "By parameters":
    c1, c2, c3 = st.columns(3)
    group_by = c1.multiselect("Group by (combine several)", params,
                              default=[p for p in ("Stage",) if p in params] or params[:1])
    color_by = c2.selectbox("Color by", ["(none)"] + params)
    show_points = c3.checkbox("Show individual samples", value=True)
    if not group_by:
        st.info("Choose at least one parameter to group by.")
        st.stop()
    plot_df = data.copy()
    plot_df["group"] = plot_df[group_by].astype(str).agg(" | ".join, axis=1)
    x_col, order = "group", sorted(plot_df["group"].unique())
elif view == "Month by month":
    if month_order is None:
        st.warning("No month information found. Add a `Month` column (name or 1-12) or a "
                   "date column to the metadata in wrangle.py and re-run it.")
        st.stop()
    c1, c2, c3 = st.columns(3)
    color_by = c1.selectbox("Split lines/boxes by", ["(none)"] + [p for p in params if p != "Month"])
    mode = c2.radio("Summary", ["Mean ± SEM", "Box plots"], horizontal=True)
    show_points = c3.checkbox("Show individual samples", value=False)
    plot_df = data.dropna(subset=["Month"]).copy()
    x_col, order = "Month", month_order
else:  # By sampling date
    if "SampleDate" not in meta.columns or meta["SampleDate"].notna().sum() == 0:
        st.warning("No usable dates found. Pick a date column in the sidebar, or check the date "
                   "format setting (Auto / Day first / Month first).")
        st.stop()
    c1, c2, c3 = st.columns(3)
    color_by = c1.selectbox("Split by", ["(none)"] + [p for p in params if p != "Month"])
    date_summary = c2.radio("Summary", ["Mean ± SEM per date", "Each sample"], horizontal=True)
    plot_df = data.dropna(subset=["SampleDate"]).copy()
    n_dates = plot_df["SampleDate"].nunique()
    show_all_dates = c3.checkbox("Label every sampling date on the axis", value=n_dates <= 40,
                                 help="Turn off if the axis gets crowded.")
    show_points = False
    x_col, order = "SampleDate", None

    dates = plot_df["SampleDate"]
    st.caption(f"Date column: {date_col} — {n_dates} distinct sampling dates, "
               f"{dates.min():%d %b %Y} to {dates.max():%d %b %Y}.")
    if (dates.dt.day == 1).all():
        st.warning("Every date in this column falls on the 1st of a month, so it only records the month, "
                   "not the day sampled. If another date column has the real sampling days, pick it in the "
                   "sidebar. Otherwise the day was lost upstream (the source metadata or wrangle.py), "
                   "and no plot can recover it.")
    n_bad = int(meta["SampleDate"].isna().sum())
    if n_bad:
        st.caption(f"{n_bad} samples have no parseable date and are left out of this view. "
                   "If many are missing, try the other date-format setting.")

color = None if color_by == "(none)" else color_by
long = plot_df.reset_index().melt(
    id_vars=[c for c in plot_df.reset_index().columns if c not in METRICS],
    value_vars=metrics, var_name="metric", value_name="value")
long["metric"] = long["metric"].map(METRICS)

if view == "By sampling date":
    hover_cols = [c for c in ("Carcass", "Stage", "Location", "Season") if c in long.columns]
    if date_summary == "Mean ± SEM per date":
        keys = ["metric", "SampleDate"] + ([color] if color else [])
        agg = long.groupby(keys, observed=True)["value"].agg(["mean", "sem", "count"]).reset_index()
        fig = px.line(agg.sort_values("SampleDate"), x="SampleDate", y="mean", error_y="sem", color=color,
                      markers=True, color_discrete_sequence=COLOR_SEQ, hover_data={"count": True})
    else:
        fig = px.scatter(long, x="SampleDate", y="value", color=color, color_discrete_sequence=COLOR_SEQ,
                         hover_data=hover_cols)
    fig.update_xaxes(type="date", title_text="Sampling date", tickformat="%d %b %Y", tickangle=-45)
    if show_all_dates:
        fig.update_xaxes(tickvals=[d.strftime("%Y-%m-%d") for d in sorted(long["SampleDate"].dropna().unique())])
elif view == "Month by month" and mode == "Mean ± SEM":
    keys = ["metric", "Month"] + ([color] if color else [])
    agg = long.groupby(keys, observed=True)["value"].agg(["mean", "sem", "count"]).reset_index()
    fig = px.line(agg, x="Month", y="mean", error_y="sem", color=color, markers=True, color_discrete_sequence=COLOR_SEQ,
                  category_orders={"Month": order},
                  hover_data={"count": True})
    if show_points:
        st.caption("Individual samples are shown in the Box plots summary.")
else:
    fig = px.box(long, x=x_col, y="value", color=color, color_discrete_sequence=COLOR_SEQ,
                 points="all" if show_points else "outliers",
                 category_orders={x_col: order})

fig.update_layout(height=550, margin=dict(t=40), template=None,
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
fig.update_yaxes(title_text=METRICS[metric])
st.plotly_chart(fig, use_container_width=True, theme="streamlit")  # follows the app theme on screen


# Download copy: same colors, plain white background, black text (for papers).
# Rendered in the browser with plotly.js, so no kaleido/Chrome install is needed.
def white_copy(figure):
    f = copy.deepcopy(figure)
    f.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(color="black", size=14), legend=dict(bgcolor="white"))
    f.update_xaxes(showline=True, linecolor="black", gridcolor="#e5e5e5",
                   title_font=dict(color="black"), tickfont=dict(color="black"))
    f.update_yaxes(showline=True, linecolor="black", gridcolor="#e5e5e5",
                   title_font=dict(color="black"), tickfont=dict(color="black"))
    return f


def download_buttons(figure, width=1100, height=550, filename="alpha_diversity"):
    spec = white_copy(figure).to_json().replace("</", "<\\/")
    components.html(f"""
    <script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>
    <style>button{{margin-right:8px;padding:6px 12px;border-radius:6px;border:1px solid #888;
    background:#fff;color:#111;cursor:pointer;font-family:sans-serif}}</style>
    <button onclick="dl('png')">Download PNG (white background)</button>
    <button onclick="dl('svg')">Download SVG (white background)</button>
    <div id="g" style="position:absolute;left:-9999px;width:{width}px;height:{height}px"></div>
    <script>
      const spec = {spec};
      async function dl(fmt) {{
        const g = document.getElementById('g');
        await Plotly.newPlot(g, spec.data, spec.layout);
        await Plotly.downloadImage(g, {{format: fmt, width: {width}, height: {height},
                                       scale: 3, filename: '{filename}'}});
      }}
    </script>""", height=50)


download_buttons(fig)

# ---------------------------------------------------------------------------
# Stats table
# ---------------------------------------------------------------------------
st.subheader("Group comparison (Kruskal-Wallis)")
rows = []
for key, label in METRICS.items():
    if key not in metrics:
        continue
    groups = [g[key].dropna().to_numpy() for _, g in plot_df.groupby(x_col) if g[key].notna().sum() > 0]
    if len(groups) < 2:
        continue
    try:
        h, p = kruskal(*groups)
    except ValueError:
        h, p = float("nan"), float("nan")
    rows.append({"metric": label, "groups": len(groups), "H": round(h, 3), "p_value": p})
if rows:
    st.dataframe(pd.DataFrame(rows).set_index("metric"), use_container_width=True)
st.caption("Kruskal-Wallis is run across the groups on the x-axis. It doesn't account for repeated "
           "sampling from the same carcass.")

with st.expander("Per-sample values"):
    st.dataframe(plot_df[metrics + [c for c in plot_df.columns if c not in METRICS]])