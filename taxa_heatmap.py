# src/taxa_heatmap.py
"""
Sample x taxon heatmaps for the dashboard.

Rows = taxa, columns = samples. Cell value is one of:
  - "relabund": plain relative abundance
  - "log10":    log10(relative abundance + half the smallest non-zero value)
                -- the best default for microbiome data, since a handful of
                dominant taxa otherwise wash out everything else
  - "clr":      centered log-ratio (computed across ALL taxa in the slice,
                then the top-N are shown -- CLR must see the whole composition)
  - "zscore":   per-taxon z-score across the displayed samples (shows *where*
                each taxon is high/low, ignoring its overall abundance)

The figure is built in dark mode (matches the dashboard);
export_utils._whiten_figure() converts it to the white-background
print version, same as the network plot.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list

TRANSFORMS = {
    "log10": "log10(relative abundance)",
    "relabund": "Relative abundance",
    "clr": "CLR (centered log-ratio)",
    "zscore": "Z-score per taxon",
}

_COLORBAR_TITLES = {
    "log10": "log10 rel. abund.",
    "relabund": "Rel. abundance",
    "clr": "CLR",
    "zscore": "Z-score",
}


def _clr(df: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    log_mat = np.log(df.to_numpy(dtype=float) + pseudocount)
    return pd.DataFrame(log_mat - log_mat.mean(axis=1, keepdims=True),
                        index=df.index, columns=df.columns)


def _cluster_order(mat: pd.DataFrame) -> list:
    """Leaf order from average-linkage clustering of the ROWS of `mat`."""
    if mat.shape[0] < 3:
        return list(mat.index)
    try:
        return list(mat.index[leaves_list(linkage(mat.to_numpy(dtype=float), "average"))])
    except Exception:
        return list(mat.index)


def prepare_matrix(abundance: pd.DataFrame, meta: pd.DataFrame = None, top_n: int = 30,
                   transform: str = "log10", cluster_samples: bool = False,
                   cluster_taxa: bool = False) -> pd.DataFrame:
    """
    `abundance`: samples (rows) x taxa (columns), relative abundance or counts.
    Returns a taxa x samples matrix, ready for plot_taxa_heatmap().
    Taxa are ranked by mean abundance and cut to `top_n` (before any
    clustering); samples are ordered by Season -> Stage -> Position
    unless `cluster_samples` is set.
    """
    df = abundance.loc[:, abundance.sum(axis=0) > 0]
    if df.shape[0] < 2 or df.shape[1] < 2:
        raise ValueError("Need at least 2 samples and 2 taxa with non-zero abundance.")

    top = df.mean(axis=0).sort_values(ascending=False).head(top_n).index

    if transform == "clr":
        sel = _clr(df)[top]
    elif transform == "log10":
        nz = df.to_numpy()[df.to_numpy() > 0]
        sel = np.log10(df[top] + nz.min() / 2)
    elif transform == "zscore":
        raw = df[top]
        sd = raw.std(axis=0, ddof=0).replace(0, np.nan)
        sel = ((raw - raw.mean(axis=0)) / sd).fillna(0)
    elif transform == "relabund":
        sel = df[top]
    else:
        raise ValueError(f"Unknown transform '{transform}'.")

    # Sample order
    if cluster_samples:
        sample_order = _cluster_order(sel)
    elif meta is not None:
        cols = [c for c in ["Season", "Stage", "Family"] if c in meta.columns]
        m = meta.loc[meta.index.intersection(sel.index)]
        sample_order = list(m.sort_values(cols + []).index) if cols else list(sel.index)
        sample_order += [s for s in sel.index if s not in sample_order]
    else:
        sample_order = list(sel.index)

    # Taxon order: by mean abundance (already sorted) unless clustering
    taxa_order = _cluster_order(sel.T) if cluster_taxa else list(top)

    return sel.loc[sample_order, taxa_order].T


def plot_taxa_heatmap(matrix: pd.DataFrame, meta: pd.DataFrame = None,
                      title: str = "Taxa x sample heatmap", transform: str = "log10") -> go.Figure:
    """Dark-theme heatmap; taxa names on y, sample IDs on x."""
    n_taxa, n_samples = matrix.shape
    diverging = transform in ("clr", "zscore")

    # Hover: sample metadata alongside the value.
    if meta is not None:
        md = meta.reindex(matrix.columns)
        season = md["Season"].astype(str).to_numpy() if "Season" in md else np.full(n_samples, "")
        stage = md["Stage"].astype(str).to_numpy() if "Stage" in md else np.full(n_samples, "")
        pos = md["Family"].astype(str).to_numpy() if "Family" in md else np.full(n_samples, "")
    else:
        season = stage = pos = np.full(n_samples, "")
    custom = np.empty((n_taxa, n_samples, 3), dtype=object)
    custom[:, :, 0] = season
    custom[:, :, 1] = stage
    custom[:, :, 2] = pos

    heat = go.Heatmap(
        z=matrix.to_numpy(dtype=float),
        x=list(matrix.columns),
        y=list(matrix.index),
        customdata=custom,
        colorscale="RdBu_r" if diverging else "Viridis",
        zmid=0 if diverging else None,
        xgap=1 if n_samples <= 60 else 0,
        ygap=1,
        colorbar=dict(title=dict(text=_COLORBAR_TITLES[transform]), thickness=16),
        hovertemplate=("<b>%{y}</b><br>Sample: %{x}"
                       "<br>Season: %{customdata[0]}  Stage: %{customdata[1]}  Position: %{customdata[2]}"
                       "<br>Value: %{z:.4g}<extra></extra>"),
        name="Abundance heatmap",
    )

    x_font = 11 if n_samples <= 30 else 9 if n_samples <= 60 else 7
    fig = go.Figure(heat)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white", size=12),
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18, color="white")),
        height=int(min(1400, max(480, 22 * n_taxa + 230))),
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis=dict(type="category", tickangle=90, tickfont=dict(size=x_font, color="white"),
                   automargin=True, title=dict(text="Sample", font=dict(color="white"))),
        yaxis=dict(type="category", autorange="reversed", tickfont=dict(size=12, color="white"),
                   automargin=True, title=dict(text="Taxon", font=dict(color="white"))),
    )
    return fig