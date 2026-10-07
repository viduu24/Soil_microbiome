# src/pages/0_Upload_Data.py
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
PROCESSED_DIR = SRC_DIR / "processed"

import streamlit as st
import pandas as pd
import plotly.express as px

import upload_pipeline as up
import data_source as ds

st.set_page_config(page_title="Upload Data", layout="wide")
st.title("Upload Your Own Dataset")
st.caption(
    "Run this same pipeline and dashboard on your own decomposition microbiome data -- "
    "upload your metadata and taxa tables, tell the app which of your own columns mean "
    "what, and every other page (Trajectories, Diversity, Differential Abundance, PMI "
    "Modeling, Source Tracking) will work on your data instead of the local example "
    "dataset. Nothing here overwrites the original local files -- this only affects "
    "what you see in THIS browser session."
)

if ds.using_uploaded_data():
    st.success("An uploaded dataset is currently active for this session.")
    if st.button("Switch back to the local example dataset"):
        ds.clear_uploaded_results()
        st.rerun()
    st.divider()

st.header("1. Upload your files")
col1, col2 = st.columns(2)
with col1:
    metadata_file = st.file_uploader(
        "Metadata file (one row per sample)", type=["csv", "txt", "tsv", "xlsx", "xls"],
        help="Needs at least: a sample ID column, a Season column, a numeric time proxy "
             "(e.g. ADH), and a decomposition Stage column. Delimiter is auto-detected "
             "for .csv/.txt/.tsv files.",
    )
with col2:
    st.write("Taxa count table(s) -- upload at least one:")
    family_taxa_file = st.file_uploader("Family-level taxa table", type=["csv", "txt", "tsv"])
    phylum_taxa_file = st.file_uploader("Phylum-level taxa table (optional)", type=["csv", "txt", "tsv"])

if metadata_file is None or (family_taxa_file is None and phylum_taxa_file is None):
    st.info("Upload a metadata file and at least one taxa table to continue.")
    st.stop()

# --- Preview + column mapping ---
st.header("2. Map your columns")
try:
    meta_preview = up.read_metadata_any_format(metadata_file)
except Exception as e:
    st.error(f"Couldn't read the metadata file: {e}")
    st.stop()

with st.expander("Preview uploaded metadata (first 5 rows)"):
    st.dataframe(meta_preview.head(), width="stretch")

meta_columns = list(meta_preview.columns)
none_option = "-- none --"

st.caption("Required:")
map_col1, map_col2 = st.columns(2)
sample_id_col = map_col1.selectbox("Sample ID column", options=meta_columns)
season_col = map_col2.selectbox("Season column", options=meta_columns)
adh_col = map_col1.selectbox("Numeric time proxy (e.g. ADH)", options=meta_columns)
stage_col = map_col2.selectbox("Decomposition Stage column", options=meta_columns)

st.caption("Optional (enable more dashboard features if provided):")
map_col3, map_col4 = st.columns(2)
position_col = map_col3.selectbox(
    "Secondary grouping (e.g. sample position)", options=[none_option] + meta_columns,
    help="Enables the same 'Shape PCoA by' / position-subset features used for Side/Under in the example dataset.",
)
carcass_col = map_col4.selectbox(
    "Individual/carcass ID column", options=[none_option] + meta_columns,
    help="Enables the Source Tracking page, which orders each individual's samples over time.",
)

column_mapping = {"sample_id": sample_id_col, "season": season_col, "adh": adh_col, "stage": stage_col}
if position_col != none_option:
    column_mapping["position"] = position_col
if carcass_col != none_option:
    column_mapping["carcass"] = carcass_col

st.caption("Optional row filter (e.g. keep only Soil samples if your file has multiple sample types):")
filter_col_choice = st.selectbox("Filter column", options=[none_option] + meta_columns)
filter_col, filter_value = None, None
if filter_col_choice != none_option:
    filter_col = filter_col_choice
    filter_value = st.selectbox("Keep rows where this column equals:", options=sorted(meta_preview[filter_col].dropna().unique().tolist()))

# --- Rarefaction depth ---
st.header("3. Rarefaction depth")
st.caption(
    "Set a target sequencing depth per rank -- samples below it are dropped, the rest "
    "are randomly subsampled to exactly this depth so all samples are compared fairly. "
    "Run once with a rough guess, check the rarefaction curve below, then adjust."
)
depth_col1, depth_col2 = st.columns(2)
rarefaction_depth = {}
if family_taxa_file is not None:
    rarefaction_depth["Family"] = depth_col1.number_input("Family rank depth", min_value=0, value=1000, step=500)
if phylum_taxa_file is not None:
    rarefaction_depth["Phylum"] = depth_col2.number_input("Phylum rank depth", min_value=0, value=1000, step=500)
rarefaction_depth = {k: (v if v > 0 else None) for k, v in rarefaction_depth.items()}

# --- Run pipeline ---
st.header("4. Run the pipeline")
if st.button("Run pipeline", type="primary"):
    try:
        with st.spinner("Processing..."):
            result = up.run_upload_pipeline(
                metadata_file=metadata_file,
                family_taxa_file=family_taxa_file,
                phylum_taxa_file=phylum_taxa_file,
                column_mapping=column_mapping,
                filter_col=filter_col,
                filter_value=filter_value,
                rarefaction_depth=rarefaction_depth,
            )
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

    ds.store_uploaded_results(
        combined_long=result["combined_long"],
        family_wide=result["family_wide"],
        phylum_wide=result["phylum_wide"],
        alpha_diversity=result["alpha_diversity"],
        core_microbiome=result["core_microbiome"],
    )
    st.session_state["upload_rarefaction_curves"] = result["rarefaction_curves"]
    st.success("Pipeline complete -- every other page now uses your uploaded data.")
    for warning in result["warnings"]:
        st.write(f"- {warning}")

if "upload_rarefaction_curves" in st.session_state:
    with st.expander("View rarefaction curves (check your depth choice)"):
        for rank_label, curve in st.session_state["upload_rarefaction_curves"].items():
            fig = px.line(curve, x="depth", y="richness", color="sample.id", height=350)
            fig.update_layout(showlegend=False, title=f"{rank_label} rarefaction curve")
            st.plotly_chart(fig, width="stretch")

# --- Optional weather merge ---
if ds.using_uploaded_data():
    st.divider()
    st.header("5. Optional: attach real calendar-day counts (DaysSinceDeath)")
    st.caption(
        "If you have a daily weather/ADH/Stage log, upload it here to attach a real "
        "elapsed-days count (and daily weather variables) to your samples, matched on "
        "Season + ADH + Stage. Skip this section if you don't have this data."
    )
    weather_file = st.file_uploader("Daily weather/ADH/Stage log", type=["csv", "txt", "tsv", "xlsx", "xls"], key="weather_upload")
    if weather_file is not None:
        try:
            weather_preview = up.read_metadata_any_format(weather_file)
        except Exception as e:
            st.error(f"Couldn't read the weather file: {e}")
            st.stop()
        with st.expander("Preview weather file (first 5 rows)"):
            st.dataframe(weather_preview.head(), width="stretch")

        weather_columns = list(weather_preview.columns)
        wcol1, wcol2 = st.columns(2)
        date_col = wcol1.selectbox("Date column", options=weather_columns, key="w_date")
        adh_col_w = wcol2.selectbox("ADH column", options=weather_columns, key="w_adh")
        stage_col_w = wcol1.selectbox("Stage column", options=weather_columns, key="w_stage")
        season_col_w = wcol2.selectbox(
            "Season column (skip if this file covers only one season)",
            options=[none_option] + weather_columns, key="w_season",
        )
        weather_mapping = {"date": date_col, "adh": adh_col_w, "stage": stage_col_w}
        if season_col_w != none_option:
            weather_mapping["season"] = season_col_w

        if st.button("Merge weather data", type="primary"):
            try:
                with st.spinner("Merging..."):
                    weather_result = up.run_upload_weather_merge(
                        combined_long=ds.get_combined_long(PROCESSED_DIR),
                        alpha_diversity=ds.get_alpha_diversity(PROCESSED_DIR),
                        weather_file=weather_file,
                        weather_column_mapping=weather_mapping,
                    )
            except Exception as e:
                st.error(f"Weather merge failed: {e}")
                st.stop()

            ds.store_uploaded_results(
                combined_long=weather_result["combined_long"],
                family_wide=ds.get_wide(PROCESSED_DIR, "Family") if family_taxa_file is not None else None,
                phylum_wide=ds.get_wide(PROCESSED_DIR, "Phylum") if phylum_taxa_file is not None else None,
                alpha_diversity=weather_result["alpha_diversity"],
            )
            st.success("Weather merge complete.")
            for warning in weather_result["warnings"]:
                st.write(f"- {warning}")
