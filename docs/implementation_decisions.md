# Implementation Specification

This document defines the computational conventions of the repository. It distinguishes parameters stated in the manuscript from additional numerical choices needed for an executable implementation.

## Censored-value estimation

Concentrations and censoring limits are represented in common ppm units during ALR calculations. The reference component is selected among strictly positive, completely uncensored elements by minimum interquartile range of log concentration. Alphabetical element order resolves exact ties. All remaining ALR coordinates are predictors for each censored target.

| Parameter or convention | Implementation | Specification basis |
|---|---|---|
| Regression engine | Random forest | Manuscript |
| Number of trees | 100 | Manuscript |
| Maximum depth | 8 | Manuscript |
| Random seed | 42 | Manuscript |
| Convergence tolerance | Relative ALR Frobenius-norm change <1e-5 | Manuscript |
| Iteration limit | 100 | Manuscript |
| Estimated cells | Source-flagged censored cells only | Manuscript |
| Initial censored values | Documented upper bounds | Additional numerical convention |
| Iteration schedule | Synchronous updates from the preceding ALR matrix | Additional numerical convention |
| Residual scale | Target-wise out-of-bag RMSE | Additional statistical convention |
| Residual-scale floor | 1e-8 ALR units | Additional numerical convention |
| Minimum observed training responses | 30 | Additional numerical convention |
| Tree sampling | Bootstrap; all predictors available at each split | Additional implementation convention |
| Minimum split / leaf sizes | 2 / 1 | Additional implementation convention |

The conditional-normal expectation below the upper ALR bound is evaluated using the Mills ratio. An asymptotic expression stabilizes extreme negative tails. Unflagged measurements are preserved, including their native-unit values in the exported concentration table. Censored estimates must be finite, positive and strictly below the supplied limits. Nonconvergence produces diagnostics without releasing a completed processed-input table. Unclassified missing or nonpositive values cause an explicit error.

The OOB scale is a single target-wise predictive-error estimate, rather than a calibrated observation-specific uncertainty. The module is an independent implementation of the published ALR/RF truncation framework. Its statistical and numerical conventions must correspond to the Methods and original processing records for exact manuscript reproduction; naming the framework alone does not establish identical software behavior.

### Artificial-censoring assessment

Complete, confirmed-observed sample rows form the eligible validation subset. Each target is evaluated separately using five 80:20 splits with seeds 42–46. Target-specific thresholds are the 10th, 20th and 30th training percentiles. Reference selection, threshold estimation and model fitting use training observations only. Hidden target values are excluded from fitting. ALR–RF, DL/2 and DL/sqrt(2) predictions are scored on identical masked test observations.

Outputs include natural-log RMSE, mean signed error, observed censoring proportions, changes in clr correlations and PCA–Procrustes residuals. Structural assessment uses up to five unscaled-clr principal components, unit Frobenius normalization and orthogonal alignment; these dimension and normalization conventions supplement the manuscript description. Test truth is used only for scoring and structural diagnostics. The assessment does not identify the true concentrations of naturally censored observations or validate simultaneous natural censoring.

## Feature construction

Six-part clr coordinates are centered by their respective medians and divided by their interquartile ranges. Zero-sum clr structure, finite transformed values and positive scales are checked. Fitted parameters are shared within each controlled comparison. Coordinate-wise scaling changes Euclidean geometry relative to unscaled Aitchison distance.

Au is converted from ppb to ppm before logarithms. Fixed component-wise unit conversion contributes constant clr-coordinate offsets that cancel under subsequent column median centering. This equivalence is covered by automated verification. Processed concentrations, rather than previously transformed coordinates, are the public command interfaces for clustering and reporting.

## Baseline and optimized clustering

K-means uses Euclidean distance, Lloyd refinement, a 300-iteration cap and tolerance 1e-4. Baselines use 30 explicit k-means++ starts per K/seed and retain the least-inertia result. Selected initial centroids are stored for refitting and stability assessment. Explicit restarts implement the stated 30-start design; exact random-number sequencing can differ from other implementations of a single n_init=30 call.

DBSCAN and OPTICS fixed baselines use min_samples=8 and radius Q95(d8). Nearest-neighbor distances count each observation itself. OPTICS uses max_eps=infinity and DBSCAN-style extraction. Fixed baselines remain reported when they fail optimized-search feasibility limits; their constraint violations and metric limitations are explicit.

The optimizers share the same feature matrix, parameter decoding, bounds, feasibility requirements, starting candidates per seed/K, search objectives and evaluation budgets. Primary feasibility requires 3–8 non-noise clusters and at least five observations per cluster; density-based methods additionally require coverage >=95%. Each run requests 416 evaluations, comprising 16 initial candidates and 25 updates of 16 candidates. Cached requests count toward the budget. For K-means, the budget applies separately to each K=3–8 and each seed.

Radius bounds use empirical quantiles with linear interpolation. Neighborhood integers use nearest-integer rounding, with round-to-even at exact halves. OPTICS neighborhood indices map to {8,10,12,15,20}. These numerical decoding conventions are held fixed across optimizers.

MOPSO uses inertia 0.72, cognitive/social coefficients 1.49, zero initial velocities, limits of 15% of the variable range and an archive capacity of 50. Clipped boundary crossings multiply the corresponding velocity by -0.5. Archive leaders use crowding tournaments. Archive truncation recalculates crowding distances after each removal. Personal-best updates and leader selection follow the documented feasibility-first rules.

NSGA-II uses binary tournaments, bounded simulated binary crossover (pair probability 0.9; distribution index 15) and polynomial mutation (per-variable probability 1/D; distribution index 20). Exact survival ties use stable candidate order; tournament ties use the run-specific random generator.

## Evaluation and configuration selection

Search objectives minimize negative silhouette estimate and DBI. The silhouette subset contains up to 600 fixed observations selected with seed 2026; density candidates use its assigned observations. DBI uses all assigned observations. Noise is excluded from assigned-only SI, DBI, CHI and DBCV and is represented separately by counts and coverage. Invalid metrics are recorded explicitly.

All evaluated feasible unique partitions undergo full-data SI/DBI reassessment. Normalization bounds pool both optimizers and all seeds within an algorithm; feasible K-means baseline candidates also enter the K-means normalization pool. Each configuration is selected from its own nondominated candidate set by squared normalized distance to the ideal. Higher coverage, larger minimum cluster size and lexicographic decoded parameters resolve subsequent ties. Objectives with zero normalization range are omitted.

CHI and DBCV are complementary evaluation metrics and do not determine the search. Different K or assigned coverage require qualified interpretation; internal validity is not independent evidence of geological significance.

## Interpretation and spatial evaluation

Elemental profiles show cluster-wise medians and interquartile ranges divided by the corresponding full-dataset elemental median. The normalization convention is defined here explicitly. Noise profiles are separate.

The interpretation-only enrichment score averages empirical percentile ranks for Cu, Au and Mo. PCA uses centered robust-scaled clr coordinates without further standardization. The nonzero-variance component with greatest absolute Spearman association to enrichment is selected and oriented positively. Explained variance resolves correlation ties. No component number is hardcoded.

C-N analysis fits continuous log–log hinge regressions with one to three segments and at least 20 distinct thresholds per segment. BIC includes breakpoint positions as parameters; accepting another segment requires an improvement of at least six units. Minimum segment sizes of 10 and 30 provide sensitivity comparisons. Differential-evolution breakpoint search is numerical and does not certify a global optimum. An epsilon-relative RSS floor limits roundoff-driven oversegmentation near exact fits.

Cluster and C-N targets are complete groups ranked by median enrichment and median Cu, respectively. Equal group scores remain complete tied groups. PCA targets follow oriented scores with exact ties retained. Noise-linked catchments contribute to the common domain but are not independently selected as clustering targets.

Linked catchments are reprojected to a documented metre-based projected CRS, clipped to the study boundary and unioned. Empty, invalid or zero-area linked geometries are rejected. Eligible occurrences are deduplicated by exact XY coordinates and counted once within or on cumulative target unions. At each 5%, 10% or 20% ceiling, the largest complete cumulative prefix that fits is evaluated. Polygons are not subdivided and lower-ranked groups are not selected by skipping an oversized prefix.

DEM hydrology, independent verification of occurrence eligibility and final map composition are external scientific/GIS operations. The software does not assign an unspecified fusion rule to the nine model maps.
