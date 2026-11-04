# Stage-by-Stage Execution

Install the package from the repository root. Use the same processed concentration artifact and configuration file for every model branch. Output directories must be new. The examples use private local inputs; none are distributed with the repository.

## 1. Common data preparation

```bash
python -m data_preparation.run --input private_data/raw_numeric.csv --censoring private_data/censoring.json --settings configs/censoring_numerics.json --output results/prepared --validate-artificial
```

The concentration CSV contains sample_id, Cu, Au, Mo, Bi, Pb and Zn; Au is ppb and the other elements ppm. Documented flags and bounds are supplied in the private censoring manifest. The output `processed_concentrations.csv` is the input to all model branches. Do not provide the separately exported scaled-clr matrix to a concentration interface.

## 2. Three baseline models

```bash
python -m baseline_models.kmeans.run --input results/prepared/processed_concentrations.csv --config configs/manuscript.json --output results/model_runs/baseline_models/kmeans
python -m baseline_models.dbscan.run --input results/prepared/processed_concentrations.csv --config configs/manuscript.json --output results/model_runs/baseline_models/dbscan
python -m baseline_models.optics.run --input results/prepared/processed_concentrations.csv --config configs/manuscript.json --output results/model_runs/baseline_models/optics
```

K-means records reference candidates across the configured K values and seeds. Density baselines record fixed reference partitions. Final selection and reporting are performed after all search branches exist.

## 3. Six optimized configurations

```bash
python -m optimized_models.kmeans.mopso.run --input results/prepared/processed_concentrations.csv --config configs/manuscript.json --output results/model_runs/optimized_models/kmeans/mopso
python -m optimized_models.kmeans.nsga_ii.run --input results/prepared/processed_concentrations.csv --config configs/manuscript.json --output results/model_runs/optimized_models/kmeans/nsga_ii
python -m optimized_models.dbscan.mopso.run --input results/prepared/processed_concentrations.csv --config configs/manuscript.json --output results/model_runs/optimized_models/dbscan/mopso
python -m optimized_models.dbscan.nsga_ii.run --input results/prepared/processed_concentrations.csv --config configs/manuscript.json --output results/model_runs/optimized_models/dbscan/nsga_ii
python -m optimized_models.optics.mopso.run --input results/prepared/processed_concentrations.csv --config configs/manuscript.json --output results/model_runs/optimized_models/optics/mopso
python -m optimized_models.optics.nsga_ii.run --input results/prepared/processed_concentrations.csv --config configs/manuscript.json --output results/model_runs/optimized_models/optics/nsga_ii
```

Each branch retains local candidate labels, decoded parameters, full-data objectives, run manifests, starting populations and request audits. Independent processes reconstruct identical starting populations from the shared seed/configuration; the joint comparison checks their actual recorded values. Candidate artifacts are source-derived private outputs and must not be uploaded publicly.

## 4. Joint configuration selection and model comparison

```bash
python -m model_comparison.run --input results/prepared/processed_concentrations.csv --config configs/manuscript.json --branches results/model_runs --output results/model_runs/model_comparison
```

Outputs include normalization bounds, selected configurations, label CSVs, validation_summary.csv and fairness_audit.json. Noise and feasibility failures are retained in reporting. The comparator does not silently relax constraints or accept mismatched branch conditions.

As an alternative to executing stages 2–4 separately, a single `python -m workflow.run` command creates the same branch/result organization and invokes comparison afterwards. Use a fresh output root for that coordinated execution.

## 5. Validation and interpretation

### Elemental profiles

```bash
python -m validation_interpretation.elemental_profiles.run --input results/prepared/processed_concentrations.csv --models results/model_runs/model_comparison --output results/interpretation/elemental_profiles
```

### PCA

```bash
python -m validation_interpretation.pca.run --input results/prepared/processed_concentrations.csv --models results/model_runs/model_comparison --output results/interpretation/pca
```

### C-N

```bash
python -m validation_interpretation.cn.run --input results/prepared/processed_concentrations.csv --models results/model_runs/model_comparison --output results/interpretation/cn
```

### Sampling stability

```bash
python -m validation_interpretation.stability.run --input results/prepared/processed_concentrations.csv --models results/model_runs/model_comparison --output results/interpretation/stability
```

### Treatment sensitivity

```bash
python -m validation_interpretation.sensitivity.run --primary-input results/prepared/processed_concentrations.csv --models results/model_runs/model_comparison --scenarios private_data/scenarios.json --output results/interpretation/sensitivity
```

The private scenario specification has a scenarios array. Each item supplies a unique name, processed_input path, scientific_basis and scaling_mode (fixed_primary or refit_common). Fixed-configuration comparisons precede any requested separate reoptimization. Add `--reoptimize-config configs/manuscript.json` to run complete independent searches for each supplied scenario afterwards.

## 6. Exploration evaluation and GIS maps

```bash
python -m exploration_evaluation.run --input results/prepared/processed_concentrations.csv --models results/model_runs/model_comparison --pca-results results/interpretation/pca --cn-results results/interpretation/cn --spatial-config private_data/spatial.json --output results/exploration
```

The spatial manifest supplies the linked catchments, study boundary, verified eligible occurrence inventory, catchment identifier field and a suitable projected CRS in metres. PCA and C-N stage manifests must match the original model-input hash. Whole rank groups, geometric unions, fixed area ceilings and unique occurrence locations determine capture. The command produces calculated tables, reference maps and capture curves, without embedding manuscript result values.
