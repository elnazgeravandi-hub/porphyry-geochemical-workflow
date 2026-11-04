# KMEANS Baseline

`model.py` contains the algorithm-specific fitting implementation. `run.py` executes only this baseline branch. Its inputs are the common positive processed-concentration artifact and shared configuration.

```bash
python -m baseline_models.kmeans.run --input results/prepared/processed_concentrations.csv --config configs/manuscript.json --output results/model_runs/baseline_models/kmeans
```

The branch stores reference candidates and their labels in private local result files. Representative selection and metric reporting occur in `model_comparison/` after all independent branches exist. This preserves the common normalization protocol.
