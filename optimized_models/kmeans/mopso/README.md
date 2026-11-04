# MOPSO–KMEANS

`model.py` defines this independent search branch; `run.py` is its executable entry point. Both optimizers use the same evaluator in `common/core.py` and numerical search operators in `common/optimizers.py`, preventing drift in objective, feasibility, decoding and budget rules.

```bash
python -m optimized_models.kmeans.mopso.run --input results/prepared/processed_concentrations.csv --config configs/manuscript.json --output results/model_runs/optimized_models/kmeans/mopso
```

Search uses shared manuscript settings. Each configured run includes initialization in its evaluation budget. K-means budgets apply separately to each K; density searches retain noise and enforce the declared coverage condition. All evaluated feasible unique partitions are retained for full-data assessment.

This branch does not select its own normalization bounds. Joint final selection is performed in `model_comparison/` using the pooled independent searches. The comparator checks actual paired starting populations, subsets, bounds, counts, source fingerprints and scientific package versions.
