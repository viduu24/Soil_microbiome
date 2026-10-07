# src/cooccurrence.py
"""
Taxon-taxon co-occurrence network, in the spirit of the co-occurrence
networks used to summarize microbial community structure in decomposition
studies (e.g. which taxa consistently rise/fall together across samples,
suggesting shared niche preference, cross-feeding, or competitive
exclusion) -- and the same idea Vidushi's earlier D3.js dashboard
prototype visualized for the Benbow lab, now reimplemented natively in
this pipeline so it shares data loading, rarefaction, and the
publication-export path (export_utils.py) with every other page.

Compositional-data caveat (same one diversity.py's ANCOM docstring
flags): relative-abundance data is constrained to sum to 1 per sample,
so a plain Pearson/Spearman correlation between two taxa's relative
abundances is biased -- an increase in one taxon mechanically forces
others down, manufacturing spurious negative correlations. This module
offers two methods:

  - method="spearman" (default): rank correlation directly on relative
    abundance. Simple and standard practice in a lot of applied
    microbiome work, but inherits the compositional bias above -- treat
    strong correlations as hypotheses, not proof, especially for
    negative edges.
  - method="clr_pearson": Pearson correlation on centered-log-ratio
    (CLR) transformed abundances (Aitchison 1982), which removes the
    sum-to-one constraint before correlating -- closer to the logic
    SparCC/SPIEC-EASI use, without pulling in those packages. This is
    the more defensible choice for anything going into a paper.

Works on a WIDE table: rows = samples (indexed by sample.id), columns =
taxa, values = relative abundance (0-1) or raw counts. Pass the
`*_relabund` columns from wrangle.py's compute_relative_abundance(), or
raw counts from diversity.py's rarefy_counts() -- CLR handles either,
since it only cares about ratios within a sample.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px


# ---------------------------------------------------------------------------
# Correlation / edge computation
# ---------------------------------------------------------------------------

def _clr_transform(wide: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """
    Centered log-ratio transform, row-wise: for each sample, subtracts
    the mean of the log-abundances across taxa. `pseudocount` avoids
    log(0) for taxa absent in a given sample -- same role as ANCOM's
    pseudocount in diversity.py.
    """
    mat = wide.to_numpy(dtype=float) + pseudocount
    log_mat = np.log(mat)
    row_means = log_mat.mean(axis=1, keepdims=True)
    clr = log_mat - row_means
    return pd.DataFrame(clr, index=wide.index, columns=wide.columns)


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Minimal BH FDR correction (matches the one in diversity.py's ANCOM)."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    bh = ranked * n / (np.arange(n) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)
    out = np.empty(n)
    out[order] = bh
    return out


def filter_by_prevalence(wide: pd.DataFrame, min_prevalence: float = 0.3) -> pd.DataFrame:
    """
    Drops taxa present (value > 0) in fewer than `min_prevalence` fraction
    of samples. Rare/mostly-absent taxa inflate the number of pairwise
    tests (weakening BH correction for everything else) and tend to
    produce unstable, sample-driven correlations -- filter them out
    before building the network, not after.
    """
    prevalence = (wide > 0).mean(axis=0)
    keep = prevalence[prevalence >= min_prevalence].index
    return wide[keep]


def compute_cooccurrence_edges(wide: pd.DataFrame, method: str = "clr_pearson",
                                min_prevalence: float = 0.3, alpha: float = 0.05,
                                min_abs_corr: float = 0.6) -> pd.DataFrame:
    """
    Computes every pairwise taxon-taxon correlation in `wide`, BH-corrects
    the p-values across all pairs, and returns edges that clear BOTH the
    significance threshold (`alpha`, on the BH-adjusted q-value) and the
    effect-size threshold (`min_abs_corr`) -- significance alone isn't a
    useful network-inclusion criterion once you have enough samples, since
    trivially small correlations become "significant".

    `method`: "clr_pearson" (recommended, see module docstring) or
    "spearman" (plain rank correlation on `wide` as given).

    Returns a DataFrame with columns: taxon_1, taxon_2, correlation,
    p_value, q_value, sign ("positive"/"negative"), sorted by |correlation|
    descending. Empty DataFrame (with the right columns) if nothing
    clears both thresholds.
    """
    filtered = filter_by_prevalence(wide, min_prevalence=min_prevalence)
    if filtered.shape[1] < 2:
        raise ValueError(
            f"Only {filtered.shape[1]} taxa passed the prevalence filter "
            f"(min_prevalence={min_prevalence}) -- lower it or check the input table."
        )

    if method == "clr_pearson":
        data = _clr_transform(filtered)
        corr_fn = pearsonr
    elif method == "spearman":
        data = filtered
        corr_fn = spearmanr
    else:
        raise ValueError(f"Unknown method '{method}' -- use 'clr_pearson' or 'spearman'.")

    taxa = data.columns.to_list()
    n_taxa = len(taxa)

    pairs, corrs, pvals = [], [], []
    for i in range(n_taxa):
        for j in range(i + 1, n_taxa):
            r, p = corr_fn(data.iloc[:, i], data.iloc[:, j])
            pairs.append((taxa[i], taxa[j]))
            corrs.append(r)
            pvals.append(p)

    pvals = np.array(pvals)
    qvals = _benjamini_hochberg(pvals) if len(pvals) > 0 else pvals

    edges = pd.DataFrame({
        "taxon_1": [p[0] for p in pairs],
        "taxon_2": [p[1] for p in pairs],
        "correlation": corrs,
        "p_value": pvals,
        "q_value": qvals,
    })

    # Publication-style effect size
    edges["r_squared"] = edges["correlation"] ** 2

    # Direction of association
    edges["sign"] = np.where(
        edges["correlation"] >= 0,
        "positive",
        "negative"
    )

    # Number of samples used for every correlation
    edges["n_samples"] = len(data)

    # Record the correlation method used
    edges["method"] = method

    # Keep only statistically significant and sufficiently strong associations
    sig = (
        (edges["q_value"] < alpha)
        & (edges["correlation"].abs() >= min_abs_corr)
    )

    # Publication-friendly column order
    edges = edges[
        [
            "taxon_1",
            "taxon_2",
            "correlation",
            "r_squared",
            "p_value",
            "q_value",
            "sign",
            "n_samples",
            "method",
        ]
    ]

    return (
        edges[sig]
        .sort_values(
            "correlation",
            key=lambda s: s.abs(),
            ascending=False
        )
        .reset_index(drop=True)
    )
# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(edges: pd.DataFrame, wide: pd.DataFrame = None) -> nx.Graph:
    """
    Builds a networkx Graph from an edges DataFrame (as returned by
    compute_cooccurrence_edges). Edge attributes: correlation, sign,
    q_value. If `wide` (the same table the edges were computed from) is
    given, each node also gets a `mean_abundance` attribute (mean across
    samples of the ORIGINAL, non-CLR values) -- useful for sizing nodes
    by overall abundance rather than by degree alone.

    Isolated taxa (no edge clears the thresholds) are never added --
    the graph only contains taxa that co-occur with at least one other.
    """
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row["taxon_1"], row["taxon_2"],
                   correlation=row["correlation"], sign=row["sign"], q_value=row["q_value"])

    if wide is not None:
        mean_abund = wide.mean(axis=0)
        for node in G.nodes:
            if node in mean_abund.index:
                G.nodes[node]["mean_abundance"] = float(mean_abund[node])

    return G


def network_layout(G: nx.Graph, seed: int = 0) -> dict:
    """Spring layout (Fruchterman-Reingold), weighted so strongly
    correlated pairs are pulled closer together -- standard for
    co-occurrence network figures."""
    weight_attr = {(u, v): abs(d["correlation"]) for u, v, d in G.edges(data=True)}
    nx.set_edge_attributes(G, weight_attr, "layout_weight")
    return nx.spring_layout(G, weight="layout_weight", seed=seed, k=None)


# ---------------------------------------------------------------------------
# Plotly figure
# ---------------------------------------------------------------------------

def plot_cooccurrence_network(
    G: nx.Graph,
    title: str = "Taxon co-occurrence network",
    label_font_size: int = 13,
) -> go.Figure:
    """
    Dark-theme figure (matches the dashboard). Every taxon is labeled.
    Labels are drawn as annotations with a semi-opaque background "halo"
    so they stay readable on top of edges, and are pushed radially outward
    from the network centre so they don't sit on their own node.
    export_utils._whiten_figure() flips all of this to a white-background
    version for downloads.
    """

    if len(G.nodes) == 0:
        raise ValueError("Cannot plot an empty network.")

    # =========================================================
    # LAYOUT
    # =========================================================

    # A moderately compact spring layout.
    # Higher k spreads nodes apart, but we don't want isolated
    # nodes flying to the corners.
    pos = nx.spring_layout(
    G,
    seed=42,
    k=5.0,
    iterations=500,
    scale=1.0,
)

    # =========================================================
    # NODE INFORMATION
    # =========================================================

    degrees = dict(G.degree())

    abundances = {
        n: float(G.nodes[n].get("mean_abundance", 0))
        for n in G.nodes
    }

    max_abundance = max(abundances.values()) if abundances else 1

    # =========================================================
    # EDGES
    # =========================================================

    positive_x = []
    positive_y = []

    negative_x = []
    negative_y = []

    for u, v, data in G.edges(data=True):

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        if data.get("sign") == "positive":
            positive_x.extend([x0, x1, None])
            positive_y.extend([y0, y1, None])
        else:
            negative_x.extend([x0, x1, None])
            negative_y.extend([y0, y1, None])

    edge_traces = []

    # Positive correlations
    if positive_x:
        edge_traces.append(
            go.Scatter(
                x=positive_x,
                y=positive_y,
                mode="lines",
                line=dict(
                    color="rgba(100,180,255,0.45)",
                    width=1.5,
                ),
                hoverinfo="none",
                name="Positive correlation",
            )
        )

    # Negative correlations
    if negative_x:
        edge_traces.append(
            go.Scatter(
                x=negative_x,
                y=negative_y,
                mode="lines",
                line=dict(
                    color="rgba(255,120,120,0.55)",
                    width=1.5,
                    dash="dash",
                ),
                hoverinfo="none",
                name="Negative correlation",
            )
        )

    # =========================================================
    # NODES
    # =========================================================

    node_x = []
    node_y = []
    node_sizes = []
    node_hover = []
    label_annotations = []

    node_colors = px.colors.qualitative.Set2

    cx = float(np.mean([p[0] for p in pos.values()]))
    cy = float(np.mean([p[1] for p in pos.values()]))

    for i, node in enumerate(G.nodes):

        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        abundance = abundances.get(node, 0)
        degree = degrees.get(node, 0)

        if max_abundance > 0:
            size = 12 + 28 * np.sqrt(abundance / max_abundance)
        else:
            size = 12
        node_sizes.append(size)

        node_hover.append(
            f"<b>{node}</b>"
            f"<br>Connections: {degree}"
            f"<br>Mean abundance: {abundance:.4f}"
        )

        # Push the label outward from the network centre, just past the
        # node's edge, anchored so text grows away from the node.
        dx, dy = x - cx, y - cy
        offset = size / 2 + 5
        if abs(dx) >= abs(dy):
            place = dict(
                xanchor="left" if dx >= 0 else "right",
                yanchor="middle",
                xshift=offset if dx >= 0 else -offset,
                yshift=0,
            )
        else:
            place = dict(
                xanchor="center",
                yanchor="bottom" if dy >= 0 else "top",
                xshift=0,
                yshift=offset if dy >= 0 else -offset,
            )

        label_annotations.append(dict(
            x=x, y=y, xref="x", yref="y",
            text=str(node),
            showarrow=False,
            font=dict(size=label_font_size, color="white"),
            bgcolor="rgba(14,17,23,0.75)",
            borderpad=2,
            **place,
        ))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=[node_colors[i % len(node_colors)] for i in range(len(node_x))],
            opacity=0.95,
            line=dict(width=1.2, color="rgba(255,255,255,0.75)"),
        ),
        name="Taxa",
    )

    # Pad the axes so labels at the edge of the layout never get clipped
    # (extra room left/right since names are wide horizontal text).
    x_min, x_max = min(node_x), max(node_x)
    y_min, y_max = min(node_y), max(node_y)
    x_pad = max(x_max - x_min, 0.5) * 0.30
    y_pad = max(y_max - y_min, 0.5) * 0.15

    # =========================================================
    # FIGURE
    # =========================================================

    fig = go.Figure()

    for trace in edge_traces:
        fig.add_trace(trace)

    fig.add_trace(node_trace)

    fig.update_layout(

        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(
                size=18,
                color="white",
            ),
        ),

        height=800,

        showlegend=True,

        hovermode="closest",

        font=dict(
            color="white",
        ),

        margin=dict(
            l=40,
            r=40,
            t=90,
            b=40,
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5,

            font=dict(
                color="white",
            ),
        ),

        annotations=label_annotations,

        xaxis=dict(
            range=[x_min - x_pad, x_max + x_pad],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
        ),

        yaxis=dict(
            range=[y_min - y_pad, y_max + y_pad],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
        ),

        template="plotly_dark",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
    )

    return fig