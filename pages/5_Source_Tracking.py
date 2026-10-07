# src/pages/4_Source_Tracking.py
import sys
from pathlib import Path
from export_utils import add_figure_download_buttons

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
PROCESSED_DIR = SRC_DIR / "processed"

import streamlit as st
import pandas as pd
import plotly.express as px

import modeling as mdl
import source_tracking as st_track
import data_source as ds

st.set_page_config(page_title="Source Tracking", layout="wide")
st.title("Source Tracking (FEAST-style)")
st.caption(
    "Estimates what fraction of each sample's community can be explained by that "
    "same carcass's community at the preceding timepoint, vs. an 'Unknown' source "
    "not explained by prior succession -- mirroring the preceding-timepoint-to-next "
    "-timepoint FEAST analysis in Kaszubinski et al. 2022 (Figure 5). This is a "
    "simplified from-scratch EM implementation, not the original R FEAST package -- "
    "see source_tracking.py for details and caveats."
)


def load_long():
    return ds.get_combined_long(PROCESSED_DIR)


if not ds.data_available(PROCESSED_DIR):
    st.error("No data available -- run wrangle.py locally, or use the Upload Data page.")
    st.stop()

df_all = load_long()
meta_cols = [c for c in df_all.columns if c not in ("taxon", "rel_abund", "rank")]

if "Carcass" not in meta_cols or "ADH" not in meta_cols:
    st.error("This page needs 'Carcass' and 'ADH' columns in your metadata to order timepoints per carcass.")
    st.stop()

st.sidebar.header("Data")
rank = st.sidebar.radio("Taxonomic rank", options=["Family", "Phylum"], horizontal=True)

wide = mdl.pivot_relabund_wide(df_all, rank=rank)  # sample.id x taxon RELATIVE abundance
# source_tracking's EM works fine on relative abundance treated as pseudo-counts
# (only the ratios among taxa matter for the multinomial mixture), so no need to
# re-derive raw counts here.
meta_lookup_cols = [c for c in meta_cols if c != "sample.id"]
meta_lookup = df_all[df_all["rank"] == rank].drop_duplicates("sample.id").set_index("sample.id")[meta_lookup_cols]

carcass_options = sorted(meta_lookup["Carcass"].dropna().unique().tolist())
selected_carcasses = st.sidebar.multiselect("Carcasses to include", options=carcass_options, default=carcass_options)

if st.sidebar.button("Run source tracking", type="primary"):
    records = []
    with st.spinner("Estimating source contributions per carcass timepoint..."):
        for carcass in selected_carcasses:
            carcass_samples = meta_lookup[meta_lookup["Carcass"] == carcass].sort_values("ADH")
            sample_order = carcass_samples.index.tolist()
            for i in range(1, len(sample_order)):
                sink_id = sample_order[i]
                source_id = sample_order[i - 1]
                if sink_id not in wide.index or source_id not in wide.index:
                    continue
                sink_counts = wide.loc[sink_id]
                source_counts = {"Preceding timepoint": wide.loc[source_id]}
                try:
                    props = st_track.estimate_source_contributions(sink_counts, source_counts)
                except ValueError:
                    continue
                for source_name, proportion in props.items():
                    records.append({
                        "sample.id": sink_id,
                        "Carcass": carcass,
                        "ADH": carcass_samples.loc[sink_id, "ADH"],
                        "source": source_name,
                        "proportion": proportion,
                    })
    st.session_state["source_result"] = pd.DataFrame.from_records(records)
    st.session_state["source_rank"] = rank

if "source_result" in st.session_state and not st.session_state["source_result"].empty:
    result = st.session_state["source_result"]
    st.subheader(f"Source contribution over time ({st.session_state['source_rank']} rank)")

    fig = px.box(
        result,
        x="ADH",
        y="proportion",
        color="source",
        points="all",
        labels={
            "proportion": "Proportion of sink community explained",
            "ADH": "ADH (sink sample)",
            "source": "Source",
        },
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )

    # Website appearance
    # Keep the normal Streamlit/Plotly background.
    fig.update_layout(
        height=500,
        margin=dict(l=80, r=40, t=40, b=70),
        font=dict(size=13),
    )

    # Display ONE graph
    st.plotly_chart(
        fig,
        width="stretch",
        key="source_tracking_box_chart",
    )

    # Download the same colored figure
    add_figure_download_buttons(
        fig,
        filename=f"source_tracking_{st.session_state['source_rank']}",
        key_prefix="source_tracking_box",
    )

    with st.expander("View underlying source-tracking data"):
        st.dataframe(result, width="stretch")
        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button("Download source-tracking data (CSV)", data=csv, file_name=f"source_tracking_{rank}.csv")
else:
    st.info("Select carcasses in the sidebar and click **Run source tracking**.")