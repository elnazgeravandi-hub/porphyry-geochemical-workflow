# Porphyry Geochemical Clustering

Computational workflow for **Hybrid Clustering and Multi-Objective Evolutionary Optimization (MOPSO–NSGA-II) for Porphyry geochemical modeling and Mineral Prospectivity Enhancement**.

The repository follows the study flowchart. Data preparation, the three baseline models, the six optimized configurations, joint comparison, interpretation and exploration evaluation have separate directories and executable entry points. Shared numerical routines are maintained in `common/` to keep optimizer comparisons consistent.

## Repository organization

| Flowchart stage | Directory | Content |
|---|---|---|
| Common data preparation | data_preparation/ | Input checks, ALR–RF censoring, clr and median–IQR scaling |
| K-means baseline | baseline_models/kmeans/ | Thirty-start k-means++ baseline and centroid refinement |
| DBSCAN baseline | baseline_models/dbscan/ | Fixed neighborhood-count and empirical-radius reference |
| OPTICS baseline | baseline_models/optics/ | Reachability ordering and DBSCAN-style baseline extraction |
| Optimized K-means | optimized_models/kmeans/mopso/ and nsga_ii/ | Independent centroid searches for K=3–8 |
| Optimized DBSCAN | optimized_models/dbscan/mopso/ and nsga_ii/ | Independent radius/neighborhood searches |
| Optimized OPTICS | optimized_models/optics/mopso/ and nsga_ii/ | Independent extraction-radius/neighborhood searches |
| Final selection and comparison | model_comparison/ | Common normalization, Pareto selection, metrics and fairness audit |
| Elemental profiles | validation_interpretation/elemental_profiles/ | Cluster medians, quartiles and scientific plots |
| PCA reference | validation_interpretation/pca/ | Data-driven component selection, orientation and plots |
| C-N reference | validation_interpretation/cn/ | Continuous regression, BIC thresholds and sensitivity |
| Sampling stability | validation_interpretation/stability/ | Twenty fixed-parameter 80% subsamples |
| Treatment sensitivity | validation_interpretation/sensitivity/ | Documented scenarios and separate optional reoptimization |
| Exploration evaluation | exploration_evaluation/ | Sample-catchment links, geometric union, occurrence capture and maps |
| Shared numerical rules | common/ | Search engines, objectives, constraints, validation and provenance |
| Coordinated execution | workflow/ | Execute independent model branches in sequence |

## Installation

Use Python 3.12 in an isolated environment. Activate the environment appropriate to the operating system, then install all scientific and verification dependencies:

```bash
python -m venv .venv
python -m pip install -e ".[report,spatial,dbcv,test]"
```

Direct dependencies are pinned in [pyproject.toml](pyproject.toml); [requirements.txt](requirements.txt) contains core dependencies. Package purposes are documented in [Python Dependencies](docs/python_dependencies.md). Runtime metadata records actual installed versions.

## Execution

All commands run from the repository root after installation. Inputs and outputs remain local. Full commands for every independent folder are given in [Execution Guide](docs/execution.md).

### Common preparation

```bash
python -m data_preparation.run --input private_data/raw_numeric.csv --censoring private_data/censoring.json --settings configs/censoring_numerics.json --output results/prepared --validate-artificial
```

### Independent model branches and joint comparison

The following coordinator runs the three separate baseline branches and six separate optimized branches, then invokes joint model comparison:

```bash
python -m workflow.run --input results/prepared/processed_concentrations.csv --config configs/manuscript.json --output results/model_runs
```

Each branch writes to its own result directory. Final selected labels, parameters and validation tables are written to `results/model_runs/model_comparison/`. The nine branches can alternatively be executed one at a time using their own `run.py` modules, as shown in the execution guide.

### Interpretation and exploration

Elemental profiles, PCA, C-N, sampling stability and exploration evaluation have separate entry points. Exploration evaluation consumes the saved PCA/C-N stages and the common model comparison; it does not substitute hardcoded manuscript table values.

## Controlled comparison

MOPSO and NSGA-II use the same input, preprocessing, search domains, initial populations, objective definitions, feasibility constraints, silhouette subset, evaluation budgets and final-selection rules within each clustering method. Primary runs use seeds 42, 142 and 242, population 16 and 416 evaluation requests per optimizer/run; the budget is applied separately to each K for K-means.

Joint comparison checks branch input/configuration hashes, sample order, preprocessing values, starting populations, search bounds, evaluation subsets and request counts. A mismatched comparison fails explicitly. Normalization is computed after pooling the independent searches, and each configuration is selected from its own nondominated candidate set. K-means baseline selection uses the shared normalization pool. Independent baseline execution therefore produces reference candidates; final representative selection belongs to the joint comparison stage.

Fixed baseline configurations are reference fits, rather than equal-budget optimizer searches. Different cluster counts and assigned coverage are reported explicitly and require qualified interpretation.

## Data access and computational scope

Original geochemical and restricted spatial records are excluded because the authors lack permission to redistribute them. See [Data Availability](DATA_AVAILABILITY.md) and [Input Schema](docs/input_schema.md).

The ALR–RF censoring module is an independently written reimplementation of the framework described by Emenaha et al. (2026; [doi:10.1007/s11004-026-10331-8](https://doi.org/10.1007/s11004-026-10331-8)). Additional numerical choices, including OOB residual-scale estimation, are explicit in [Implementation Specification](docs/implementation_decisions.md). Matching these conventions to the actual study analysis is necessary for exact numerical reproduction.

Software tests use artificial fixtures and verify algorithmic behavior, independent branch execution, comparison consistency and stage integration. They do not establish numerical agreement with the restricted original manuscript dataset. DEM delineation, independent geological interpretation and final GIS composition are external scientific operations. The software does not invent an unspecified fusion of the nine maps.

## Verification

```bash
python -m pytest tests -q
```

See [Software Verification](docs/validation.md) for the recorded verification scope and result.
