# Synthetic workflow demonstration

This example generates fully synthetic geochemical and spatial inputs and executes the existing research modules. No original study records, coordinates, fitted parameters or manuscript result values are used to generate the example.

## Scope

The example tests execution and computational consistency. It does not reproduce the manuscript's numerical results, establish geological validity or demonstrate optimizer superiority. Original geochemical and spatial inputs remain subject to data-owner restrictions. Exact numerical reproduction requires authorized access to those inputs.

The generator creates 180 six-element observations in three artificial populations, documented Au/Bi censoring flags, abstract overlapping square polygons and artificial occurrence points. Squares stand in for spatial supports only: they are not hydrologically delineated catchments. The metre-based coordinate frame is a numerical test frame, not a representation of the study area. One duplicate occurrence deliberately tests deduplication. Known synthetic concentrations are retained separately and are not supplied to the primary censoring estimator.

## Installation and execution

From the repository root, activate a Python environment and install the project:

```bash
python -m pip install -e ".[report,spatial,dbcv,test]"
python examples/synthetic_demo/run.py --output results/synthetic_demo --artificial-validation
```

Use a new output directory for each run. Existing results are never overwritten. Omit `--artificial-validation` to omit only the additional five-split artificial-censoring assessment. Preparation, all nine models, comparison, profiles, PCA, C-N, sampling stability, fixed-configuration treatment sensitivity and GIS evaluation still run.

The demonstration retains the manuscript populations of 16, 25 updates, 416 requests per search run, K=3-8, seeds 42/142/242, objectives, search-domain definitions, constraints and selection rule. Only the expected input count is changed from 2,715 to 180 and the dataset is synthetic. Both optimizers receive the same generated input and paired starting populations. This is a complete-budget demonstration and may take several minutes or longer depending on hardware; it is not a shortcut benchmark.

For the documented treatment sensitivity example, only artificially censored values are replaced by DL/2; selected configurations and primary preprocessing are held fixed. Separate sensitivity reoptimization remains available through the existing sensitivity interface and is not run by this example. DEM delineation and geological interpretation are outside this numerical demonstration.

## Expected outputs and acceptance checks

| Relative output | Expected content |
|---|---|
| inputs/synthetic_provenance.json | Generator seed, independent origin and scope |
| prepared/processed_concentrations.csv | Positive processed concentrations; observed measurements preserved |
| prepared/artificial_validation_summary.csv | Five-split artificial-censoring assessment when requested |
| model_runs/model_comparison/validation_summary.csv | Nine configurations: three baselines and six optimized branches |
| model_runs/model_comparison/fairness_audit.json | 24 paired comparisons; 416 requests per optimizer in every pair |
| interpretation/ | Profiles, PCA, C-N, 20-subsample stability and DL/2 sensitivity |
| exploration/occurrence_capture.csv | 33 rows: nine clustering models and two references at three area ceilings |
| exploration/reference_maps.png | Maps of the artificial supports |
| verification.json | Acceptance checks written only after successful completion |

The runner exits with an error if a subprocess fails, a selected optimized configuration is absent, budgets differ or expected table dimensions fail. Infeasible configurations are not replaced or silently relaxed. The core modules additionally check input hashes, matching starts, preprocessing and branch conditions. Exact floating-point values or byte-identical figures across platforms are not an acceptance criterion. Use the pinned dependencies and inspect the recorded environment when comparing runs.

Generated inputs and outputs are created locally under `results/` and remain ignored by Git. The public example contains the deterministic generator and runner, not any private data. Optional output plots should be identified as synthetic whenever shared.

## Verified reference run

[verified_reference.json](verified_reference.json) records a completed run including artificial-censoring validation, the nine model summaries, the 33 spatial comparisons and the actual scientific package versions. It is an illustrative reference, not a numerical target to tune against. Acceptance checks are structural and numerical-validity checks rather than exact equality to stored scores. Some optimizers recover identical partitions on these deliberately simple populations; this is a valid outcome. The existing 41-test suite also passed after adding this example.
