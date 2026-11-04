# Local Output Organization

Model branches write outputs under the same structure used for their source directories. These are private derived artifacts and are not included in the public repository.

| Directory under results/model_runs/ | Outputs |
|---|---|
| baseline_models/kmeans/ | Reference candidates across configured K/seeds |
| baseline_models/dbscan/ | Fixed DBSCAN reference partition |
| baseline_models/optics/ | Fixed OPTICS reference partition |
| optimized_models/kmeans/mopso/ | MOPSO centroid-search candidates and audits |
| optimized_models/kmeans/nsga_ii/ | NSGA-II centroid-search candidates and audits |
| optimized_models/dbscan/mopso/ | MOPSO DBSCAN candidates and audits |
| optimized_models/dbscan/nsga_ii/ | NSGA-II DBSCAN candidates and audits |
| optimized_models/optics/mopso/ | MOPSO extraction candidates and audits |
| optimized_models/optics/nsga_ii/ | NSGA-II extraction candidates and audits |
| model_comparison/ | Selected parameters/labels, shared normalization, internal metrics and fairness audit |

Every branch has candidates.json, candidate_labels.npz and branch_manifest.json. Search branches additionally have per-K/per-seed request and audit files. NPZ loading disables pickle. The joint comparator validates bundle identity, input/configuration hashes, sample order, preprocessing, source fingerprint and scientific package versions. An optimizer with no feasible candidate retains an explicit infeasible outcome; fixed density baselines remain reported without relaxing optimized-search constraints.

Interpretation and exploration result paths are specified independently through --output. PCA and C-N stage manifests bind their scores to the exact original model input. Exploration evaluation reads those saved outputs instead of fitting a new reference under different conditions.
