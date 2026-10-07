# src/alpha_diversity_page.py
"""
Streamlit page: alpha diversity (observed richness, Chao1, Shannon) on
rarefied counts, viewable by any combination of metadata parameters or
month by month. Computation comes from diversity.py; this file is UI only.

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
# from the first column whose name contains "date" (case-insensitive).

METRICS = {"observed_richness": "Observed richness", "chao1": "Chao1", "shannon": "Shannon"}
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


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


def add_month(meta: pd.DataFrame):
    """Adds a 'Month' column (ordered chronologically). Returns (meta, ordered_labels)
    or (meta, None) if no month information exists."""
    if "Month" in meta.columns:
        raw = meta["Month"]
        if pd.api.types.is_numeric_dtype(raw):
            num = raw.astype("Int64")
        else:
            lookup = {m.lower(): i + 1 for i, m in enumerate(MONTH_ABBR)}
            num = raw.astype(str).str[:3].str.lower().map(lookup).astype("Int64")
        meta = meta.assign(Month=num.map(lambda n: MONTH_ABBR[n - 1] if pd.notna(n) else None))
        return meta, [m for m in MONTH_ABBR if m in set(meta["Month"].dropna())]

    date_col = next((c for c in meta.columns if "date" in str(c).lower()), None)
    if date_col is None:
        return meta, None
    dt = pd.to_datetime(meta[date_col], errors="coerce")
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

alpha = compute_alpha(df, depth)
date_like = [c for c in df.columns if "date" in str(c).lower()]
meta = (df[[SAMPLE_COL] + [c for c in META_COLS + ["Month"] + date_like if c in df.columns]]
        .drop_duplicates(SAMPLE_COL).set_index(SAMPLE_COL))
meta = meta.rename(columns={"Family": "Location"})
meta, month_order = add_month(meta)
data = alpha.join(meta, how="inner")

params = [c for c in meta.columns if c != "ADH" and c not in date_like]
if not metrics or not params:
    st.info("Pick at least one metric.")
    st.stop()

view = st.radio("View", ["By parameters", "Month by month"], horizontal=True)

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
else:
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

color = None if color_by == "(none)" else color_by
long = plot_df.reset_index().melt(
    id_vars=[c for c in plot_df.reset_index().columns if c not in METRICS],
    value_vars=metrics, var_name="metric", value_name="value")
long["metric"] = long["metric"].map(METRICS)

if view == "Month by month" and mode == "Mean ± SEM":
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