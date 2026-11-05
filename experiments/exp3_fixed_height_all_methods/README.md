# Fixed-height, all-methods study: code

This folder holds the code for the round: the setup files and the three notebooks.

```
fixed_height_all_methods/
├── setup/       setup_pipeline.py, fh_pipeline.py, pf_infer_timed.py  (used by every notebook)
└── notebooks/   1_Generate_Trees_and_Sequences  (CPU)
                 2_Run_Phyloformer               (GPU)
                 3_Run_Other_Methods             (CPU)
```

The notebooks write their output to their own folders, not here:
- `simulated_data/fixed_height_all_methods/`  trees (true and inferred) and sequences
- `results/fixed_height_all_methods/`         results CSVs and failures_*.csv

Run order: notebook 1 first, then 2 and 3 (they can run at the same time).
Set `CLOCK_RATE` to your own rate in each notebook. Earlier rounds are not touched.
The Cherry model data is read from `CherryML_data(CoreCode)/` at the repo root.
