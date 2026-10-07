# check_metadata_columns.py
# One-off diagnostic: run this to see whether your metadata already has a
# column distinguishing stillborn vs. adult carcasses, and if not, to list
# every distinct Carcass ID used in the Summer season so you can identify
# the split yourself from your field records.
#
# Usage: edit META_PATH below to match your real metadata file, then run:
#   python check_metadata_columns.py

import pandas as pd

META_PATH = "C:/Users/vidus/OneDrive/Documents/Desktop/Benbow RA/Re_ Datasets/meta-ch3-new2.txt"

meta = pd.read_csv(META_PATH, sep="\t")

print("=== All columns in your metadata ===")
print(list(meta.columns))

print("\n=== Looking for anything that might indicate carcass age/type ===")
candidate_keywords = ["age", "type", "still", "adult", "class", "group", "size"]
matches = [c for c in meta.columns if any(k in c.lower() for k in candidate_keywords)]
if matches:
    print(f"Found possible match(es): {matches}")
    for c in matches:
        print(f"\nUnique values in '{c}':", meta[c].dropna().unique().tolist())
else:
    print("No obviously-named column found. You'll need to check field records "
          "to know which Carcass IDs were stillborn vs. adult.")

if "Order" in meta.columns and "Season" in meta.columns:
    summer_soil = meta[(meta["Order"].str.strip().str.lower() == "soil") & (meta["Season"] == "Summer")]
    print("\n=== Distinct Carcass IDs used in Summer (Soil samples only) ===")
    print(sorted(summer_soil["Carcass"].dropna().unique().tolist()))
    print(f"\nTotal distinct Summer carcasses: {summer_soil['Carcass'].nunique()}")