# Soil microbiome decomposition pipeline + dashboard

## What's new in this update

**`wrangle.py`** (edit the file paths at the top of `run_pipeline()` back to
your real `meta-ch3-new2.txt` / `Family_taxa_table.csv` / `Phylum_taxa_table.csv`
paths before running):

- Condenses duplicate-named taxa columns (unchanged from before).
- **New: rarefies** each rank's count table to a fixed sequencing depth
  (`RAREFACTION_DEPTH` near the top of the file, default 7000/7000) before
  computing relative abundance, dropping any sample below that depth.
  Also writes `processed/family_rarefaction_curve.parquet` and
  `processed/phylum_rarefaction_curve.parquet` -- **open these and check
  where each sample's richness curve plateaus before trusting the default
  depth.** Kaszubinski et al. 2022 picked 7000 (16S)/4000 (ITS) this way,
  not by guessing. If your depth drops too many samples, lower it; if
  curves haven't plateaued at your chosen depth, raise it.
- **New:** exports `processed/alpha_diversity.parquet` (observed richness,
  Chao1, Shannon per sample/rank, joined to your metadata) and
  `processed/core_microbiome.parquet` (taxa present in 100% of a Season's
  samples, by default -- change `group_col`/`presence_threshold` in the
  `core_microbiome()` call in `run_pipeline()` if you want Stage instead,
  or a lower prevalence cutoff).
- `family_wide.parquet` / `phylum_wide.parquet` / `combined_long.parquet`
  are still produced, now from the rarefied counts.

**New modules** (import these directly, or use them through the dashboard):

- `diversity.py` -- rarefaction curve/rarefaction, alpha diversity
  (richness/Chao1/Shannon), beta diversity (Bray-Curtis/Jaccard) + PCoA,
  PERMANOVA, and **ANCOM** (compositionally-correct differential
  abundance testing -- this is what the papers use instead of a plain
  t-test on relative abundances).
- `modeling.py` -- random forest regression of community composition
  against ADH/collection date (cross-validated, with feature importances),
  and quadratic succession-curve fitting for individual taxa (mirrors
  Figure 3 in Kaszubinski et al. 2022).
- `source_tracking.py` -- a from-scratch EM implementation of the FEAST
  idea (percent of a sample's community explained by a prior timepoint's
  community, vs. "Unknown"). **Not** the original R FEAST package --
  treat it as directionally informative, not publication-grade, if you
  need to match the papers' exact numbers.

**Dashboard** is now multi-page. `trajectories.py` is still the entry point
(unchanged) with a new `pages/` folder next to it:

- `pages/1_Diversity.py` -- alpha diversity by group, beta diversity PCoA
  + PERMANOVA.
- `pages/2_Differential_Abundance.py` -- run ANCOM across Season/Stage/Order.
- `pages/3_PMSI_Modeling.py` -- random forest regression + quadratic taxon
  trajectories.
- `pages/4_Source_Tracking.py` -- preceding-timepoint source contribution
  per carcass, boxplotted over ADH.

## Running it

```bash
pip install -r requirements.txt
python wrangle.py          # regenerates processed/*.parquet with rarefaction
streamlit run trajectories.py
```

Streamlit will auto-discover the `pages/` folder and show all five pages in
the sidebar nav. Every page checks for its required `processed/*.parquet`
file and tells you to run `wrangle.py` first if it's missing.

## Testing note

All five pages (including every interactive button: ANCOM run, random
forest fit, PERMANOVA, source tracking) were exercised end-to-end with
Streamlit's `AppTest` harness against synthetic data shaped like your real
tables (metadata + duplicate-column taxa tables) before delivery, so the
code paths are confirmed to execute without exceptions. That's not a
substitute for running it against your actual `Family_taxa_table.csv` /
`Phylum_taxa_table.csv` -- real data can still surprise you (e.g. a
metadata column with unexpected NaNs), so treat first-run output with a
skeptical eye, especially the ANCOM and source-tracking pages.

## Known simplifications vs. the papers

- Beta diversity uses Bray-Curtis/Jaccard, not (un)weighted UniFrac --
  UniFrac needs a phylogenetic tree, which your Family/Phylum-level
  pre-aggregated tables don't carry. If you have ASV-level data with a
  tree, that would be a more faithful match to the papers.
- ANCOM here is a direct from-scratch reimplementation of the Mandal et
  al. 2015 algorithm (log-ratio Kruskal-Wallis + BH correction), not a
  call to the R `ancom.R`/`ANCOMBC` package -- results should be very
  similar but not bit-for-bit identical.
- Source tracking is a simplified EM point-estimate, not the full FEAST
  Bayesian/EM procedure from the R package.
- Rarefaction here is done directly on the Family/Phylum-aggregated count
  tables (since that's what you have), not on ASV-level counts before
  aggregation. This is a reasonable practical approximation but means
  Family-rank and Phylum-rank samples are rarefied independently and may
  end up representing very slightly different underlying read subsamples.
