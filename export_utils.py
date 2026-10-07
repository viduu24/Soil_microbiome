# src/export_utils.py
"""
Publication-ready figure export for the dashboard.

Every plot in this project is a Plotly figure. Plotly's own camera icon
exports a low-res, transparent-background PNG, which is *not* what a
journal wants -- most require a flat white (or fully opaque) background
and either a specific DPI or a lossless/TIFF format for print.

This module gives every dashboard page a one-line way to add "Download
PNG" / "Download TIFF" buttons that produce figures ready to drop
directly into a manuscript:

  - White (fully opaque) background, both around the plot AND behind the
    legend/paper area (Plotly's `paper_bgcolor` and `plot_bgcolor` are
    forced to white regardless of the app's own theme).
  - High resolution via Plotly/Kaleido's `scale` factor (default scale=4
    against a 1000x700 base figure = 4000x2800 px, comfortably above the
    ~300 DPI most journals ask for at typical print sizes).
  - TIFF export at an explicit DPI tag (default 300) using LZW
    (lossless) compression, which is what most journals mean by
    "TIFF" -- this is a real DPI-tagged TIFF, not a renamed PNG.

Requires the `kaleido` package (for Plotly -> static image rendering)
and `Pillow` (for the PNG -> TIFF conversion + DPI tagging). Both are in
requirements.txt.

Usage inside any dashboard page, right after building `fig`:

    from export_utils import add_figure_download_buttons
    st.plotly_chart(fig, use_container_width=True)
    add_figure_download_buttons(fig, filename="pcoa_by_season")

That renders two `st.download_button`s side by side. If you need the
raw bytes instead (e.g. to bundle several figures into one zip), use
`fig_to_png_bytes` / `fig_to_tiff_bytes` directly.
"""

import io
import copy

import plotly.graph_objects as go
import streamlit as st
from PIL import Image


def _whiten_figure(fig: go.Figure) -> go.Figure:
    """Create a white-background export without changing the screen figure."""
    fig2 = copy.deepcopy(fig)

    fig2.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        title_font=dict(color="black"),
        legend=dict(
            bgcolor="white",
            font=dict(color="black"),
            title_font=dict(color="black"),
        ),
    )

    # Override white text explicitly set on network node traces.
    # Node and edge colors remain unchanged.
    for trace in fig2.data:
        if hasattr(trace, "textfont"):
            trace.update(textfont=dict(color="black"))

    # Taxon labels on the network are annotations: black text on a
    # near-opaque white halo so they stay readable over edges.
    fig2.update_annotations(font=dict(color="black"), bgcolor="rgba(255,255,255,0.85)")

    # Network edge/node colors tuned for white: the dark-mode pastels are
    # too faint on white paper.
    for trace in fig2.data:
        name = getattr(trace, "name", None)
        if name == "Positive correlation":
            trace.update(line=dict(color="rgba(20,90,200,0.6)"))
        elif name == "Negative correlation":
            trace.update(line=dict(color="rgba(200,30,30,0.65)"))
        elif name == "Taxa":
            trace.update(marker=dict(line=dict(color="rgba(40,40,40,0.85)")))

    fig2.update_xaxes(
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
    )
    fig2.update_yaxes(
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
    )

    return fig2


def fig_to_png_bytes(fig: go.Figure, width: int = 1000, height: int = 700, scale: int = 4) -> bytes:
    """
    Renders `fig` to PNG bytes with a forced white background.

    `scale` multiplies width/height for the actual rendered pixel
    dimensions (Plotly/Kaleido convention) -- e.g. width=1000, scale=4
    renders at 4000px wide. Journals commonly ask for ~300 DPI at the
    figure's intended print size (e.g. a 3.5" single-column figure at
    300 DPI is 1050px); the default here (4000x2800) comfortably covers
    single- or double-column placement at 300+ DPI without re-exporting.
    """
    fig2 = _whiten_figure(fig)
    return fig2.to_image(format="png", width=width, height=height, scale=scale)


def fig_to_tiff_bytes(fig: go.Figure, width: int = 1000, height: int = 700, scale: int = 4,
                       dpi: int = 300, compression: str = "tiff_lzw") -> bytes:
    """
    Renders `fig` to TIFF bytes with a forced white background and an
    explicit DPI tag (most journals check the DPI metadata, not just
    pixel count). Uses LZW compression by default -- lossless, and the
    format most submission systems expect for "TIFF, LZW compression".

    Implementation note: Kaleido/Plotly can't write TIFF directly, so
    this renders a high-res PNG first (see fig_to_png_bytes) and
    re-encodes it as TIFF via Pillow, which is where the DPI tag and
    compression are actually applied.
    """
    png_bytes = fig_to_png_bytes(fig, width=width, height=height, scale=scale)
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")  # drop alpha -> guarantees opaque white, not just visually white
    buf = io.BytesIO()
    img.save(buf, format="TIFF", dpi=(dpi, dpi), compression=compression)
    return buf.getvalue()


def add_figure_download_buttons(fig: go.Figure, filename: str, key_prefix: str = None,
                                 width: int = 1000, height: int = 700, scale: int = 4, dpi: int = 300,
                                 lazy: bool = True) -> None:
    """
    Renders download buttons (PNG, TIFF) for `fig`, both white-background
    and print-resolution. Call this immediately after
    `st.plotly_chart(fig, ...)` on any dashboard page.

    `filename` should be a bare name with no extension, e.g.
    "alpha_diversity_by_season" -- ".png"/".tiff" are appended.
    `key_prefix` disambiguates Streamlit widget keys when the same page
    calls this more than once (e.g. one plot per tab/rank) -- defaults
    to `filename` if not given.

    `lazy=True` (default) puts both exports behind a single "Prepare
    downloads" button, so Kaleido only renders (the slow step, ~1-2s per
    image) when the researcher actually wants a file -- important on
    pages with several figures, where eagerly rendering every one on
    every Streamlit rerun would make the page noticeably sluggish.
    Set `lazy=False` for a one-off page where that cost doesn't matter
    and you'd rather skip the extra click.
    """
    key_prefix = key_prefix or filename

    def _render_buttons():
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download PNG",
                data=fig_to_png_bytes(fig, width=width, height=height, scale=scale),
                file_name=f"{filename}.png",
                mime="image/png",
                key=f"{key_prefix}_png_dl",
            )
        with col2:
            st.download_button(
                "Download TIFF",
                data=fig_to_tiff_bytes(fig, width=width, height=height, scale=scale, dpi=dpi),
                file_name=f"{filename}.tiff",
                mime="image/tiff",
                key=f"{key_prefix}_tiff_dl",
            )

    if not lazy:
        _render_buttons()
        return

    if st.button("Prepare downloads (PNG / TIFF, white background, print-res)", key=f"{key_prefix}_prep_btn"):
        with st.spinner("Rendering print-resolution images..."):
            _render_buttons()